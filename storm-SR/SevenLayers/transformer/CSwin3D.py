# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, patches_time, T_sp, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.patches_time = patches_time
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.T_sp = T_sp
        self.get_v = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def video2cswin(self, x):
        B, N, C = x.shape

        H = W = self.resolution
        T = self.patches_time
        x = x.transpose(-2,-1).contiguous().view(B, C, T, H, W)
        x = video2windows(x, self.T_sp, self.H_sp, self.W_sp)
        # [-1, T_sp*H_sp* W_sp, C]
        x = x.reshape(-1, self.T_sp*self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # [-1, num_heads, T_sp* H_sp* W_sp, C//num_heads]
        return x

    def get_lepe(self, x, func):
        H = W = self.resolution
        T = self.patches_time

        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, T, H, W)

        T_sp, H_sp, W_sp = self.T_sp, self.H_sp, self.W_sp
        x = x.view(B, C, T//T_sp, T_sp, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().reshape(-1, C, T_sp, H_sp, W_sp) # B', C, H', W'

        lepe = func(x) # B', C, T', H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, T_sp*H_sp*W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, T_sp*self.H_sp*self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Video2Window
        H = W = self.resolution
        T = self.patches_time
        B, L, C = q.shape
        assert L == H * W * T, "flatten img_tokens has wrong size"
        
        q = self.video2cswin(q)
        # [-1, num_heads, T_sp* H_sp* W_sp, C//num_heads]
        k = self.video2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe # B head N N @ B head N C
        # B head N C//head -> B N head C//head -> B N C
        x = x.transpose(1, 2).reshape(-1, self.T_sp * self.H_sp * self.W_sp, C)

        # Window2Video
        x = windows2video(x, self.T_sp, self.H_sp, self.W_sp, T, H, W).view(B, -1, C)  # B H' W' C

        return x

class CSWin3DBlock(nn.Module):
    def __init__(self, dim, reso, patches_time, num_heads,
                 split_size=7, T_sp=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.3,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.pathes_time = patches_time
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.T_sp = T_sp

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, patches_time=patches_time, T_sp=self.T_sp, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, patches_time=self.pathes_time, T_sp=self.T_sp, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, T*H*W, C
        """
        H = W = self.patches_resolution
        T = self.pathes_time
        B, L, C = x.shape
        assert L == H * W * T, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        # B,L,C
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        # 3,B,L,C

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
            # B,L,C
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def video2windows(video, T_sp, H_sp, W_sp):
    """
    video: B C T H W
    """
    B, C, T, H, W = video.shape
    video_reshape = video.view(B, C, T//T_sp, T_sp, H // H_sp, H_sp, W // W_sp, W_sp)
    # [B, T//T_sp, H // H_sp, W // W_sp, T_sp, H_sp, W_sp, C]
    video_perm = video_reshape.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, T_sp * H_sp* W_sp, C)
    # [-1, H_sp* W_sp, C]
    return video_perm

def windows2video(video_splits_hw, T_sp, H_sp, W_sp, T, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(video_splits_hw.shape[0] / (H * W * T / H_sp / W_sp / T_sp))
    video = video_splits_hw.view(B, T // T_sp, H // H_sp, W // W_sp, T_sp, H_sp, W_sp, -1)
    video = video.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, -1)
    return video

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_time, img_size=224, embed_dim=96, depth=[2,2], split_size=[2, 4],
                 num_heads=[8, 12], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.3, hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        # self.use_chk = use_chk
        # self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        # self.stage1_conv_embed = nn.Sequential(
        #     nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
        #     Rearrange('b c h w -> b (h w) c', h = img_size//4, w = img_size//4),
        #     nn.LayerNorm(embed_dim)
        # )

        self.before_trans = nn.Sequential(
            # nn.Conv3d(in_chans, embed_dim, 3, 1, 1),
            Rearrange('b c s h w -> b (s h w) c'),
            nn.LayerNorm(embed_dim)
        )

        self.after_trans = nn.Sequential(
            # nn.Conv3d(in_chans, embed_dim, 3, 1, 1),
            nn.LayerNorm(embed_dim),
            Rearrange('b (s h w) c -> b c s h w', s=img_time, h=img_size, w=img_size)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWin3DBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size, patches_time=img_time, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        # self.merge1 = Merge_Block(curr_dim, curr_dim*2)
        # curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList(
            [CSWin3DBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size, patches_time=img_time, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer)
            for i in range(depth[1])])
        
        # self.merge2 = Merge_Block(curr_dim, curr_dim*2)
        # curr_dim = curr_dim*2
        # temp_stage3 = []
        # temp_stage3.extend(
        #     [CSWinBlock(
        #         dim=curr_dim, num_heads=heads[2], reso=img_size//16, mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
        #         drop=drop_rate, attn_drop=attn_drop_rate,
        #         drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer)
        #     for i in range(depth[2])])

        # self.stage3 = nn.ModuleList(temp_stage3)
        #
        # self.merge3 = Merge_Block(curr_dim, curr_dim*2)
        # curr_dim = curr_dim*2
        # self.stage4 = nn.ModuleList(
        #     [CSWinBlock(
        #         dim=curr_dim, num_heads=heads[3], reso=img_size//32, mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
        #         drop=drop_rate, attn_drop=attn_drop_rate,
        #         drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True)
        #     for i in range(depth[-1])])
       
        # self.norm = norm_layer(curr_dim)
        # Classifier head
        # self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print ('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        # B = x.shape[0]
        # x = self.stage1_conv_embed(x)
        x = self.before_trans(x)

        for blk in self.stage1:
            x = blk(x)

        for blk in self.stage2:
            x = blk(x)

        # for blk in self.stage1:
        #     if self.use_chk:
        #         x = checkpoint.checkpoint(blk, x)
        #     else:
        #         x = blk(x)
        # for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
        #                        [self.stage2, self.stage3, self.stage4]):
        #     x = pre(x)
        #     for blk in blocks:
        #         if self.use_chk:
        #             x = checkpoint.checkpoint(blk, x)
        #         else:
        #             x = blk(x)
        # x = self.norm(x)
        # return torch.mean(x, dim=1)
        x = self.after_trans(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

### 224 models

@register_model
def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1,2,21,1],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

### 384 models

@register_model
def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model

@register_model
def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model

