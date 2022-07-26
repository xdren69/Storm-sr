import torch
from torch import nn
from SevenLayers.transformer.general_transformer import TransformerModel, LearnedPositionalEncoding


class TimeTransformer(nn.Module):
    def __init__(
            self,
            seq_length,
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            input_dropout_rate,
            attn_dropout_rate,
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
        x = self.position_encoding(x)
        x = self.pre_dropout(x)
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        return x


class TimeSpaceTransformer(nn.Module):
    def __init__(self):
        super(TimeSpaceTransformer, self).__init__()
        self.timeTrans = None
        self.spaceTrans = None