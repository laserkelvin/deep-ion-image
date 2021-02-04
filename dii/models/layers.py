import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout


class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel,
        dropout=0.0,
        activation=nn.LeakyReLU(0.3),
        pool=nn.MaxPool2d(2),
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel, **kwargs)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.pool = pool

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.conv(X)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)
        output = self.pool(output)
        return output


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dropout=0.0,
        activation=nn.ReLU,
        upsample_size=2,
        batch_norm=True,
        reflection=1,
        padding=0,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        self.dropout = nn.Dropout(dropout)
        self.upsample_size = upsample_size
        self.reflection_pad = nn.ReflectionPad2d(reflection)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = F.interpolate(
            X, scale_factor=self.upsample_size, mode="bilinear", align_corners=False
        )
        output = self.reflection_pad(output)
        output = self.conv(output)
        if self.norm is not None:
            output = self.norm(output)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)
        return output


class TransposeDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel,
        dropout=0.0,
        activation=nn.LeakyReLU(0.3),
        batch_norm=True,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, conv_kernel, **kwargs)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.conv(X)
        if self.norm is not None:
            output = self.norm(output)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        num_channels,
        use_1x1conv=False,
        kernel_size=3,
        stride=1,
        activation=nn.ReLU,
        pool: int = 0,
        upsample: int = 0,
        dropout: float = 0.,
    ):
        super().__init__()
        layers = []
        if upsample != 0:
            layers.append(nn.Upsample(scale_factor=upsample, mode="bilinear"))
        layers.extend(
            [
                nn.Conv2d(
                    input_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=1,
                    stride=stride,
                ),
                nn.Dropout(dropout),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=1,
                ),
                nn.BatchNorm2d(num_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)
        if use_1x1conv:
            self.skip = nn.Conv2d(input_channels, num_channels, 1, stride=stride)
        else:
            self.skip = None
        # if activation provided, instantiate the layer
        if activation:
            self.activation = activation()
        if pool != 0:
            self.pool = nn.MaxPool2d(pool)
        else:
            self.pool = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.conv(X)
        if self.skip:
            X = self.skip(X)
            Y.add_(X)
        if self.activation:
            Y = self.activation(Y)
        if self.pool:
            Y = self.pool(Y)
        return Y
