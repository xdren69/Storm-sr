import torch
import torch.nn as nn
from torch.nn import functional as F


class SevenLayers(nn.Module):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1,
            num_layers=0
    ):
        super(SevenLayers, self).__init__()
        self.img_dim = img_dim
        self.img_time = img_time
        self.f_maps = f_maps
        self.trans_f_maps = f_maps[0] if isinstance(f_maps, list) else f_maps
        self.init_conv = InitConv(in_channels=in_channel, out_channels=self.trans_f_maps, dropout=input_dropout_rate)
        self.encoders = self.generate_Encoder(
            f_maps=f_maps,
            num_layers=num_layers
        )
        if isinstance(f_maps, list):
            f_maps.reverse()
        self.decoders = self.generate_Decoder(
            f_maps=f_maps,
            num_layers=num_layers
        )
        if isinstance(f_maps, list):
            f_maps.reverse()
        self.final_conv = nn.Conv3d(self.trans_f_maps, in_channel, 1)

    def generate_Encoder(self, f_maps, num_layers=0):
        model_list = nn.ModuleList([])
        if isinstance(f_maps, list):
            for idx in range(1, len(f_maps)):
                encoder_layer = EncoderLayer(
                    in_channels=f_maps[idx-1],
                    out_channels=f_maps[idx],
                )
                model_list.append(encoder_layer)
        else:
            for idx in range(num_layers):
                encoder_layer = ResidualLayer(
                    channels=f_maps
                )
                model_list.append(encoder_layer)
        return model_list

    def generate_Decoder(self, f_maps, num_layers):
        model_list = nn.ModuleList([])
        if isinstance(f_maps, list):
            for idx in range(1, len(f_maps)):
                decoder_layer = DecoderLayer(
                    in_channels=f_maps[idx-1],
                    out_channels=f_maps[idx],
                )
                model_list.append(decoder_layer)
        else:
            for idx in range(num_layers):
                encoder_layer = ResidualLayer(
                    channels=f_maps
                )
                model_list.append(encoder_layer)
        return model_list

    def process_by_trans(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):
        x = self.init_conv(x)

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        x = self.process_by_trans(x)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features)

        pred = self.final_conv(x)
        return pred


class SingleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1
    ):
        super(SingleConv, self).__init__()
        self.add_module('Conv3d',
                        nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
        self.add_module('ReLU', nn.ReLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            if_encoder,
            kernel_size=3
    ):
        super(DoubleConv, self).__init__()
        if if_encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, padding=1))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, padding=1))


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)
        return y


class ResidualLayer(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size=3
    ):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, encoder_features=None):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out += identity
        out = self.relu2(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3
    ):
        super(EncoderLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=True
        )

    def forward(self, x):
        x = self.conv_net(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3
    ):
        super(DecoderLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels*2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=False
        )

    def forward(self, x, encoder_features):
        # use nearest neighbor interpolation and concatenation joining
        # output_size = encoder_features.size()[2:]
        # x = F.interpolate(x, size=output_size, mode='nearest')
        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        x = torch.cat((encoder_features, x), dim=1)

        x = self.conv_net(x)
        return x
