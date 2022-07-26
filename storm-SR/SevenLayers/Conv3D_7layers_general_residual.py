from SevenLayers.SevenLayers import SevenLayers, DoubleConv, ResidualLayer


class Conv3d_7layers_general_residual(SevenLayers):
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
            input_dropout_rate=0.1,
            num_layers=0
    ):
        super(Conv3d_7layers_general_residual, self).__init__(
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            f_maps=f_maps,
            input_dropout_rate=input_dropout_rate,
            num_layers=num_layers
        )

        self.middle_conv = ResidualLayer(
            channels=f_maps
        )

    def process_by_trans(self, x):
        return self.middle_conv(x)

    def forward(self, x):
        x = self.init_conv(x)
        identity = x

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        x = self.process_by_trans(x)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features)

        x = x + identity
        pred = self.final_conv(x)
        return pred
