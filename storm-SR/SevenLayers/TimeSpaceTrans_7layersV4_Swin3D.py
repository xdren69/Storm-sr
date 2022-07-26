import torch
import torch.nn as nn
from einops import rearrange
from SevenLayers.transformer.TimeSpaceTransformer import TimeSpaceTransformer_elsa
from SevenLayers.transformer.Swin3D import SwinTransformer3D
from SevenLayers.SevenLayersV2 import SevenLayersV2, SingleConv
from .utils import initialize_weights


class TimeSpaceTrans_7layers_Swin3D(SevenLayersV2):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            depths=[2, 2, 6, 2],
            window_size=(2, 7, 7),
            num_heads=[3, 6, 12, 24],
            drop_path_rate=0.2,
            mlp_ratio=4.,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1
    ):
        super(TimeSpaceTrans_7layers_Swin3D, self).__init__(
            img_dim,
            img_time,
            in_channel,
            f_maps=f_maps,
            input_dropout_rate=input_dropout_rate
        )

        self.img_time = img_time
        self.img_size = img_dim
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

        self.trans = SwinTransformer3D(
            in_chans=in_channel,
            embed_dim=embedding_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=drop_path_rate,
        )

        self.apply(initialize_weights)


    def process_by_trans(self, x):
        x = self.conv_before_trans(x)
        x = self.trans(x)
        x = self.conv_after_trans(x)
        return x
