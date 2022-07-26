import torch
import torch.nn as nn
from SevenLayers.transformer.SwinTransformer_without_downSample import SwinTransformer3D
from SevenLayers.SevenLayers import SevenLayers, SingleConv


class Trans_7layers_swin(SevenLayers):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            hidden_ration=4.,
            path_dropout_rate=0.2,
            attn_dropout_rate=0.,
            f_maps=[16, 32, 64],
            trans_depth=[2, 2, 2, 2],
            trans_heads=[3, 6, 12, 24],
            window_size=[4, 4, 4],
            patch_size=[4, 4, 4],
            input_dropout_rate=0.1
    ):
        super(Trans_7layers_swin, self).__init__(
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            f_maps=f_maps,
            input_dropout_rate=input_dropout_rate
        )

        self.img_time = img_time
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim

        self.conv_before_trans = SingleConv(
            f_maps[-1],
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_after_trans = SingleConv(
            embedding_dim,
            f_maps[-1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.transformer = SwinTransformer3D(
            patch_size=patch_size,
            in_chans=f_maps[-1],
            embed_dim=embedding_dim,
            depths=trans_depth,
            num_heads=trans_heads,
            window_size=window_size,
            mlp_ratio=hidden_ration,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=attn_dropout_rate,
            drop_path_rate=path_dropout_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=False,
            frozen_stages=-1,
        )

    def process_by_trans(self, x):
        x = self.conv_before_trans(x)
        x = self.transformer(x)
        x = self.conv_after_trans(x)
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #1x

    def forward(self, x):
        position_embeddings = self.position_embeddings
        return x + position_embeddings