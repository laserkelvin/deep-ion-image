
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from dii.models.layers import ConvolutionBlock, DecoderBlock


class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = ConvolutionBlock(1, 8, 11, pool=nn.MaxPool2d(4))
        self.conv_2 = ConvolutionBlock(8, 16, 7)
        self.conv_3 = ConvolutionBlock(16, 32, 5, pool=nn.MaxPool2d(4))
        self.conv_4 = ConvolutionBlock(32, 64, 3)
        self.conv_5 = ConvolutionBlock(64, 128, 3, pool=nn.MaxPool2d(4))
        self.conv_6 = ConvolutionBlock(128, 256, 1, activation=None)
        self.layers = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.conv_6]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.conv_1(X)
        for layer in self.layers[1:]:
            output = layer(output)
        return output


class BaseDecoder(nn.Module):
    def __init__(self, sigmoid=True):
        super().__init__()
        self.conv_1 = DecoderBlock(256, 128, 5, upsample_size=8)
        self.conv_2 = DecoderBlock(128, 64, 7, upsample_size=8)
        self.conv_3 = DecoderBlock(64, 32, 5, upsample_size=6)
        self.conv_4 = DecoderBlock(32, 16, 3, upsample_size=2)
        self.conv_5 = DecoderBlock(16, 8, 3, upsample_size=2)
        self.conv_6 = DecoderBlock(8, 1, 3, upsample_size=2)
        self.conv_7 = DecoderBlock(1, 1, 3,  activation=nn.Sigmoid(), upsample_size=1)
        self.layers = [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5, self.conv_6, self.conv_7]

    def forward(self, X: torch.Tensor):
        output = self.conv_1(X)
        for layer in self.layers[1:]:
            output = layer(output)
        return output


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, Y = batch
        pred_Y = self.forward(X)
        loss = F.binary_cross_entropy(pred_Y, Y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}


class TransposeDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel, dropout=0., activation=nn.ReLU(), batch_norm=True, **kwargs):
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


class TransposeDecoder(nn.Module):
    def __init__(self, sigmoid=True):
        super().__init__()
        self.model = nn.Sequential(
            TransposeDecoderBlock(256, 128, 5, stride=4, padding=1, output_padding=3),
            TransposeDecoderBlock(128, 64, 2, stride=4, padding=1, output_padding=2),
            TransposeDecoderBlock(64, 32, 4, stride=3, padding=1, output_padding=2),
            TransposeDecoderBlock(32, 16, 3, stride=3, padding=1, output_padding=1),
            TransposeDecoderBlock(16, 8, 2, stride=3, padding=0, output_padding=1),
            TransposeDecoderBlock(8, 1, 2, stride=2, padding=0, batch_norm=False, activation=nn.Sigmoid())
        )

    def forward(self, X: torch.Tensor):
        return self.model(X)
