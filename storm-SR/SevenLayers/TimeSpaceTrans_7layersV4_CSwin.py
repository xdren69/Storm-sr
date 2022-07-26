import torch
import torch.nn as nn
from einops import rearrange
from SevenLayers.transformer.TimeSpaceTransformer import TimeSpaceTransformer_elsa
from SevenLayers.transformer.CSwin3D import CSWinTransformer
from SevenLayers.SevenLayersV2 import SevenLayersV2, SingleConv
from .utils import initialize_weights


class TimeSpaceTrans_7layers_CSwin(SevenLayersV2):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            # window_size,
            num_heads,
            # hidden_dim,
            # num_transBlock,
            # attn_dropout_rate,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1
    ):
        super(TimeSpaceTrans_7layers_CSwin, self).__init__(
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

        self.trans = CSWinTransformer(
            img_size=self.img_size,
            img_time=img_time//(2**len(f_maps)),
            embed_dim=embedding_dim,
            num_heads=num_heads,
            depth=[2, 2],
            split_size=[2, 4]
        )

        # self.apply(initialize_weights)


    def process_by_trans(self, x):
        x = self.conv_before_trans(x)
        x = self.trans(x)
        x = self.conv_after_trans(x)
        return x
