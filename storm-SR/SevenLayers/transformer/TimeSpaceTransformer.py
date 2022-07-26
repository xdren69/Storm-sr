import torch
from torch import nn
from einops import rearrange
from SevenLayers.transformer.Transformer import TransformerModel, LearnedPositionalEncoding
from SevenLayers.transformer.SwinTransformer2D import BasicLayer
from SevenLayers.transformer.NeighborhoodAttention import NeighborhoodAttention, Mlp
from SevenLayers.transformer.elsa.elsa import ELSABlock
from timm.models.layers import DropPath


class TimeTransformer(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            input_dropout_rate,
            attn_dropout_rate
    ):
        super(TimeTransformer, self).__init__()
        self.transformer = TransformerModel(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=hidden_dim,
            dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, seq_length
        )
        self.pre_dropout = nn.Dropout(p=input_dropout_rate)
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b h w) s c')
        x = self.position_encoding(x)
        x = self.pre_dropout(x)
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = rearrange(x, '(b p1 p2) s c -> b c s p1 p2', p1=H, p2=W)
        return x

class TimeTransformer_v2(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            input_dropout_rate,
            attn_dropout_rate,
            path_dropout_rate=0.0
    ):
        super(TimeTransformer_v2, self).__init__()
        self.transformer = TransformerModel(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=hidden_dim,
            dropout_rate=path_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, seq_length
        )
        self.pre_dropout = nn.Dropout(p=input_dropout_rate)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b h w) s c')
        x = self.position_encoding(x)
        x = self.pre_dropout(x)
        x = self.transformer(x)
        x = rearrange(x, '(b p1 p2) s c -> b c s p1 p2', p1=H, p2=W)
        return x

class SpaceTransformer_nat(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            hidden_size,
            kernel_size=7,
            # mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_scale=None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_size, act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):

        B, C, D, H, W = x.size()

        x = rearrange(x, 'b c s h w -> (b s) h w c')
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(self.gamma1 * x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        x = rearrange(x, '(b p1) h w c -> b c p1 h w', p1=D)
        return x


class SpaceTransformer_elsa(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            hidden_size,
            kernel_size=7,
            # mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            group_width=4,
            layer_scale=None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size

        self.elsa_block = ELSABlock(
            dim=dim, num_heads=num_heads,
            dim_qk=None, dim_v=None, kernel_size=kernel_size,
            stride=1, dilation=1,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, group_width=group_width,
            groups=1, lam=1, gamma=1
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b s) h w c')
        x = self.elsa_block(x)
        x = rearrange(x, '(b p1) h w c -> b c p1 h w', p1=D)
        return x


class SpaceTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_layers,
            num_heads,
            window_size,
            hidden_dim,
            attn_drop=0.,
            input_drop=0.,
    ):
        super(SpaceTransformer, self).__init__()
        self.transformer = BasicLayer(
            dim=embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=hidden_dim/embedding_dim,
            qkv_bias=True,
            qk_scale=None,
            drop=input_drop,
            attn_drop=attn_drop,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b s) (h w) c')
        x = self.transformer(x, H, W)
        x = rearrange(x, '(b p1) (h p2) c -> b c p1 h p2', p1=D, p2=W)
        return x


class TimeSpaceTransformer(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            space_window_size,
            attn_dropout_rate,
            input_dropout_rate,
            num_time_trans_layer=2,
            num_space_trans_layer=2,
    ):
        super(TimeSpaceTransformer, self).__init__()
        self.timeTrans = TimeTransformer(
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            num_layers=num_time_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            input_dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.spaceTrans = SpaceTransformer(
            embedding_dim=embedding_dim,
            num_layers=num_space_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            window_size=space_window_size,
            attn_drop=attn_dropout_rate,
        )

    def forward(self, x):
        x_after_time = self.timeTrans(x)
        x = x + x_after_time
        x_after_space = self.spaceTrans(x)
        return x_after_space

class TimeSpaceTransformer_v2(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            space_window_size,
            attn_dropout_rate,
            input_dropout_rate,
            num_time_trans_layer=2,
            num_space_trans_layer=2,
    ):
        super(TimeSpaceTransformer_v2, self).__init__()
        self.timeTrans = TimeTransformer_v2(
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            num_layers=num_time_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            input_dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            path_dropout_rate=0.0
        )
        self.spaceTrans = SpaceTransformer(
            embedding_dim=embedding_dim,
            num_layers=num_space_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            window_size=space_window_size,
            attn_drop=attn_dropout_rate,
        )

    def forward(self, x):
        x = self.timeTrans(x)
        x = self.spaceTrans(x)
        return x


class TimeSpaceTransformer_nat(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            space_window_size,
            attn_dropout_rate,
            input_dropout_rate,
            num_time_trans_layer=1,
            num_space_trans_layer=2,
    ):
        super(TimeSpaceTransformer_nat, self).__init__()
        self.timeTrans = TimeTransformer(
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            num_layers=num_time_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            input_dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.spaceTrans = SpaceTransformer_nat(
            dim=embedding_dim,
            num_heads=num_heads,
            hidden_size=hidden_dim,
            kernel_size=space_window_size,
            attn_drop=attn_dropout_rate,
        )

    def forward(self, x):
        x_after_time = self.timeTrans(x)
        x = x + x_after_time
        x_after_space = self.spaceTrans(x)
        return x + x_after_space


class TimeSpaceTransformer_elsa(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            space_window_size,
            attn_dropout_rate,
            input_dropout_rate,
            num_time_trans_layer=1,
            num_space_trans_layer=2,
    ):
        super(TimeSpaceTransformer_elsa, self).__init__()
        self.timeTrans = TimeTransformer(
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            num_layers=num_time_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            input_dropout_rate=input_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.spaceTrans = SpaceTransformer(
            embedding_dim=embedding_dim,
            num_layers=num_space_trans_layer,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            window_size=space_window_size,
            attn_drop=attn_dropout_rate,
        )
        self.elsaTrans = SpaceTransformer_elsa(
            dim=embedding_dim,
            num_heads=num_heads,
            hidden_size=hidden_dim,
            kernel_size=space_window_size,
        )

    def forward(self, x):
        x = self.timeTrans(x)
        x = self.elsaTrans(x)
        x = self.spaceTrans(x)
        return x
