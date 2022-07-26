import torch
import torch.nn as nn
from SevenLayers.transformer.SwinTransformer3D import DecoderBasicLayer, EncoderBasicLayer, PatchMerging, PatchSpliting, MiddelBasicLayer
from SevenLayers.SevenLayers import SevenLayers, SingleConv, DoubleConv


class Trans_7layers_swin_Encode(SevenLayers):
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
        self.img_time = img_time
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.depths = trans_depth
        self.window_size = window_size
        self.mlp_ratio = hidden_ration
        self.qkv_bias = True
        self.qk_scale = None
        self.attn_dropout_rate = attn_dropout_rate
        self.drop_path_rate = path_dropout_rate
        self.num_heads = trans_heads

        super(Trans_7layers_swin_Encode, self).__init__(
            img_dim,
            img_time,
            in_channel,
            embedding_dim,
            f_maps=f_maps,
            input_dropout_rate=input_dropout_rate
        )

        self.init_conv = DoubleConv(
            in_channels=in_channel,
            out_channels=embedding_dim,
            if_encoder=True
        )

        self.final_conv = DoubleConv(
            in_channels=embedding_dim,
            out_channels=in_channel,
            if_encoder=False
        )

    def generate_Encoder(self, f_maps):
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        num_layers = len(self.depths)
        model_list = nn.ModuleList([])
        for i_layer in range(num_layers):
            layer = EncoderBasicLayer(
                dim=int(self.embedding_dim * 2 ** i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=0.,
                attn_drop=self.attn_dropout_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging,
                use_checkpoint=False)
            model_list.append(layer)
        return model_list

    def generate_Decoder(self, f_maps):
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        num_layers = len(self.depths)
        model_list = nn.ModuleList([])
        for i_layer in range(num_layers):
            layer_idx = num_layers - i_layer - 1
            layer = DecoderBasicLayer(
                dim=int(self.embedding_dim * 2 ** layer_idx)*2,
                depth=self.depths[layer_idx],
                num_heads=self.num_heads[layer_idx],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                if_first=True if layer_idx==num_layers-1 else False,
                drop=0.,
                attn_drop=self.attn_dropout_rate,
                drop_path=dpr[sum(self.depths[layer_idx:]):sum(self.depths[layer_idx:])],
                norm_layer=nn.LayerNorm,
                upsample=PatchSpliting,
                use_checkpoint=False)
            model_list.append(layer)
        return model_list

    def process_by_trans(self, x):
        return x