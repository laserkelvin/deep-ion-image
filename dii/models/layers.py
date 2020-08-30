import torch
from torch import nn
from torch.nn import functional as F


class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel,
        dropout=0.0,
        activation=nn.LeakyReLU(0.3, inplace=True),
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
        conv_kernel,
        dropout=0.0,
        activation=nn.LeakyReLU(0.3, inplace=True),
        upsample_size=2,
        batch_norm=True,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel, **kwargs)
        self.activation = activation
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        self.dropout = nn.Dropout(dropout)
        self.upsample_size = upsample_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = F.interpolate(
            X, scale_factor=self.upsample_size, mode="bilinear", align_corners=True
        )
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
        activation=nn.LeakyReLU(0.3, inplace=True),
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
