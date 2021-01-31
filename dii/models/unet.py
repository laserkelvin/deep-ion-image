import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, final_act=nn.Sigmoid()):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = self._make_encoder(n_channels, bilinear)
        factor = 2 if bilinear else 1
        self.down4 = UnetEncoderBlock(512, 1024 // factor)
        self.up1 = UnetDecoderBlock(1024, 512 // factor, bilinear)
        self.up2 = UnetDecoderBlock(512, 256 // factor, bilinear)
        self.up3 = UnetDecoderBlock(256, 128 // factor, bilinear)
        self.up4 = UnetDecoderBlock(128, 64, bilinear)
        self.outc = OutputLayer(64, n_classes, final_act)

    @staticmethod
    def _make_encoder(in_channels: int, bilinear: bool = True) -> nn.Module:
        chan_sizes = [in_channels, 64, 128, 256, 512]
        factor = 2 if bilinear else 1
        chan_sizes.append(1024 // factor)
        modules = list()
        for index, chan in enumerate(chan_sizes):
            if index == 0:
                layer = DoubleConv
            else:
                layer = UnetEncoderBlock
            try:
                modules.append(layer(chan, chan_sizes[index + 1]))
            except IndexError:
                pass
        encoder = nn.Sequential(
            modules
        )
        return encoder

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # upsampling from the encoding
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = F.relu(self.up4(x, x1))
        output = self.outc(x)
        return output


class SkimUNet(UNet):
    def __init__(self, n_channels, n_classes, bilinear=True, final_act=nn.Sigmoid()):
        super().__init__(n_channels, n_classes, bilinear, final_act)

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = UnetEncoderBlock(16, 32)
        self.down2 = UnetEncoderBlock(32, 64)
        self.down3 = UnetEncoderBlock(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = UnetEncoderBlock(128, 256 // factor)
        self.up1 = UnetDecoderBlock(256, 128 // factor, bilinear)
        self.up2 = UnetDecoderBlock(128, 64 // factor, bilinear)
        self.up3 = UnetDecoderBlock(64, 32 // factor, bilinear)
        self.up4 = UnetDecoderBlock(32, 16, bilinear)
        self.outc = OutputLayer(16, n_classes, final_act)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UnetEncoderBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )
        valid_activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.3),
            "prelu": nn.PReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU()
        }
        if activation and activation in valid_activations:
            self.activation = valid_activations.get(activation)
        else:
            self.activation = None

    def forward(self, x):
        x = self.maxpool_conv(x)
        if self.activation:
            x = self.activation(x)
        return x


class UnetDecoderBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, activation=None):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
        valid_activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.3),
            "prelu": nn.PReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU()
        }
        if activation and activation in valid_activations:
            self.activation = valid_activations.get(activation)
        else:
            self.activation = None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.Sigmoid()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        output = self.conv(x)
        if self.activation:
            output = self.activation(output)
        return output


class UnetEncoder(nn.Module):
    def __init__(self, n_channels, start_features=16, bilinear=True, activation=None):
        super().__init__()
        self.inc = DoubleConv(n_channels, start_features)
        self.down1 = UnetEncoderBlock(start_features, start_features * 2, activation)
        self.down2 = UnetEncoderBlock(start_features * 2, start_features * 4, activation)
        self.down3 = UnetEncoderBlock(start_features * 4, start_features * 8, activation)
        factor = 2 if bilinear else 1
        self.down4 = UnetEncoderBlock(start_features * 8, (start_features * 16) // factor, activation)

    def forward(self, X: torch.Tensor):
        x1 = self.inc(X)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class UnetDecoder(nn.Module):
    def __init__(self, n_channels, n_segs=9, start_features=16, bilinear=True, activation=None, final_act=nn.Sigmoid()):
        super().__init__()
        factor = 2 if bilinear else 1
        self.up1 = UnetDecoderBlock(start_features * 16, (start_features * 8) // factor, bilinear, activation)
        self.up2 = UnetDecoderBlock(start_features * 8, (start_features * 4) // factor, bilinear, activation)
        self.up3 = UnetDecoderBlock(start_features * 4, (start_features * 2) // factor, bilinear, activation)
        self.up4 = UnetDecoderBlock(start_features * 2, start_features, bilinear, activation=nn.ReLU())
        self.outc = OutputLayer(n_segs + 1, n_channels, None)
        # this layer will generate masks for each image
        self.seg = OutputLayer(start_features, n_segs + 1, None)

    def forward(self, outputs):
        # unpack the residuals from encoder
        x1, x2, x3, x4, x5 = outputs
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = F.relu(self.up4(x, x1))
        mask = self.seg(x)
        output = self.outc(mask)
        return output, mask
