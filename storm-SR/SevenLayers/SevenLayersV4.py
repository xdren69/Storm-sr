import torch
import torch.nn as nn
from torch.nn import functional as F


class SevenLayersV4(nn.Module):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1,
            num_layers=0
    ):
        super(SevenLayersV4, self).__init__()
        self.img_dim = img_dim
        self.img_time = img_time
        self.f_maps = f_maps
        # 2Conv + Down
        self.encoders = self.generate_Encoder(
            f_maps=[in_channel] + f_maps
        )

        # up + 2Conv
        self.decoders = self.generate_Decoder(
            f_maps=f_maps[::-1] + [in_channel]
        )

    def generate_Encoder(self, f_maps, num_layers=0):
        model_list = nn.ModuleList([])


        for idx in range(1, len(f_maps)):

            if idx == 1:
                encoder_layer = EncoderLayer(
                    in_channels=f_maps[idx - 1],
                    out_channels=f_maps[idx],
                    down_sample_type="None"
                )
            elif idx == 2:
                encoder_layer = EncoderLayer(
                    in_channels=f_maps[idx - 1],
                    out_channels=f_maps[idx],
                    down_sample_type="unet"
                )
            else:
                encoder_layer = EncoderLayer(
                    in_channels=f_maps[idx-1],
                    out_channels=f_maps[idx],
                    down_sample_type="time"
                )

            model_list.append(encoder_layer)

        return model_list

    def generate_Decoder(self, f_maps):
        model_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            if idx == len(f_maps)-1:
                decoder_layer = DecoderLayer(
                    in_channels=f_maps[idx-1],
                    out_channels=f_maps[idx],
                    up_sample_type="None"
                )
            elif idx == len(f_maps)-2:
                decoder_layer = DecoderLayer(
                    in_channels=f_maps[idx - 1],
                    out_channels=f_maps[idx],
                    up_sample_type="unet"
                )
            else:
                decoder_layer = DecoderLayer(
                    in_channels=f_maps[idx - 1],
                    out_channels=f_maps[idx],
                    up_sample_type="time"
                )
            model_list.append(decoder_layer)
        return model_list

    def process_by_trans(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):


        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        x = self.process_by_trans(x)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features)

        return x


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
            down_sample_type="time",
            kernel_size=3
    ):
        super(EncoderLayer, self).__init__()
        self.down_sample_type = down_sample_type
        if down_sample_type == "time":
            self.down_sample = nn.Conv3d(in_channels, in_channels, kernel_size=(3,3,3), stride=(2,1,1), padding=(1,1,1))
        elif down_sample_type == "unet":
            self.down_sample = nn.Conv3d(in_channels, in_channels, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        elif down_sample_type == "None":
            pass
        else:
            raise NotImplementedError

        self.conv_net = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=True
        )

    def forward(self, x):
        if self.down_sample_type != "None":
            x = self.down_sample(x)
        x = self.conv_net(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            up_sample_type="time",
            kernel_size=3,
    ):
        super(DecoderLayer, self).__init__()
        self.conv_net = DoubleConv(
            in_channels=in_channels*2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            if_encoder=False
        )
        self.up_sample_type = up_sample_type
        if up_sample_type == "time":
            self.up_sample = nn.ConvTranspose3d(in_channels=in_channels*2, out_channels=in_channels*2,
                                                kernel_size=(4,3,3), stride=(2,1,1), padding=(1,1,1))
        elif up_sample_type == "unet":
            self.up_sample = nn.ConvTranspose3d(in_channels=in_channels*2, out_channels=in_channels*2,
                                                kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=1)
        elif up_sample_type == "None":
            pass
        else:
            raise NotImplementedError

    def forward(self, x, encoder_features):
        # x += encoder_features
        x = torch.cat((encoder_features, x), dim=1)
        if self.up_sample_type != "None":
            x = self.up_sample(x)
        x = self.conv_net(x)
        return x
