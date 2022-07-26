import torch
import torch.nn as nn
from Unet.UnetBlocks import Encoder, FinalConv, DoubleConv
from SevenLayers.SevenLayersV2 import SingleConv
from Unet.utils import create_feature_maps
from torch.nn import functional as F
from SevenLayers.transformer.CSwin3D import CSWinTransformer

class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            upSample,
            kernel_size=3,
            scale_factor=(2, 2, 2),
            basic_module=DoubleConv,
            conv_layer_order='cr',
            num_groups=8
    ):
        super(Decoder, self).__init__()
        self.upSample = upSample
        if self.upSample == "ConvT":
            self.upSampler = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=1,
                output_padding=1
            )
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upSample == "ConvT":
            x = self.upsample(x)
            x += encoder_features
        elif self.upSample == "Inter":
            x = torch.cat((encoder_features, x), dim=1)
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        elif self.upSample == None:
            x = torch.cat((encoder_features, x), dim=1)
        else:
            raise NotImplemented

        x = self.basic_module(x)
        return x

class UNet3DCswin(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            img_time,
            img_size,
            trans_dim=96,
            f_maps=64,
            layer_order='cr',
            num_groups=8,
            **kwargs
    ):
        super(UNet3DCswin, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=3)
            # [16, 32, 64, 128]

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.before_trans = SingleConv(
            f_maps[-1],
            trans_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.trans = CSWinTransformer(
            img_size=img_size//(2**(len(f_maps)-1)),
            img_time=img_time//(2**(len(f_maps)-1)),
            embed_dim=trans_dim,
            num_heads=[4, 8],
            depth=[2, 2],
            split_size=[2, 4]
        )
        self.after_trans = SingleConv(
            trans_dim,
            f_maps[-1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps)-1):
            in_feature_num = reversed_f_maps[i]*2
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(
                in_feature_num, out_feature_num,
                upSample="Inter",
                basic_module=DoubleConv,
                conv_layer_order=layer_order, num_groups=num_groups
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        # encoders_features = encoders_features[1:]
        x = self.before_trans(x)
        x = self.trans(x)
        x = self.after_trans(x)

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        return x