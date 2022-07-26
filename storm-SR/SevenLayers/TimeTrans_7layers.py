import torch
import torch.nn as nn
from einops import rearrange
from SevenLayers.transformer.Transformer import TransformerModel
from SevenLayers.SevenLayers import SevenLayers, SingleConv


class TimeTrans_7layers(SevenLayers):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            num_trans_layers,
            num_heads,
            hidden_dim,
            trans_dropout_rate,
            attn_dropout_rate,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1
    ):
        super(TimeTrans_7layers, self).__init__(
            img_dim,
            img_time,
            in_channel,
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

        self.transformer = TransformerModel(
            dim=embedding_dim,
            depth=num_trans_layers,
            heads=num_heads,
            mlp_dim=hidden_dim,
            dropout_rate=trans_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )

        self.seq_length = self.img_time
        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, self.seq_length
        )
        self.pre_dropout = nn.Dropout(p=trans_dropout_rate)
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

    def process_by_trans(self, x):
        x = self.conv_before_trans(x)
        B, C, D, H, W = x.size()
        x = rearrange(x, 'b c s h w -> (b h w) s c')
        x = self.position_encoding(x)
        x = self.pre_dropout(x)
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = rearrange(x, '(b p1 p2) s c -> b c s p1 p2', p1=H, p2=W)
        x = self.conv_after_trans(x)
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #1x

    def forward(self, x):
        position_embeddings = self.position_embeddings
        return x + position_embeddings