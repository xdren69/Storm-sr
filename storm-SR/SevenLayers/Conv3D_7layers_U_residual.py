from SevenLayers.SevenLayers import SevenLayers, DoubleConv


class Conv3d_7layers_U_residual(SevenLayers):
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
        super(Conv3d_7layers_U_residual, self).__init__(
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            f_maps=f_maps,
            input_dropout_rate=input_dropout_rate
        )

        self.middle_conv = DoubleConv(
            in_channels=f_maps[-1],
            out_channels=f_maps[-1],
            if_encoder=False
        )

    def process_by_trans(self, x):
        return self.middle_conv(x)
