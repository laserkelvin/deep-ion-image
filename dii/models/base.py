
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

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        pred_Y = self.forward(X)
        loss = F.binary_cross_entropy(pred_Y, Y)
        tensorboard_logs = {"validation_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
