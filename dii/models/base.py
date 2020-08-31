from typing import Dict, List

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F

from dii.models import layers
from dii.models.unet import UNet, SkimUNet


def initialize_weights(m):
    for name, parameter in m.named_parameters():
        if "norm" not in name:
            try:
                nn.init.kaiming_uniform_(parameter, nonlinearity="leaky_relu")
            except:
                pass


class BaseEncoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            layers.ConvolutionBlock(1, 4, 3, padding=1),
            layers.ConvolutionBlock(4, 8, 3, padding=1),
            layers.ConvolutionBlock(8, 16, 3, padding=1),
            layers.ConvolutionBlock(16, 32, 3, padding=1),
            layers.ConvolutionBlock(32, 48, 3, padding=1),
            layers.ConvolutionBlock(48, 64, 3, padding=1),
            layers.ConvolutionBlock(64, 72, 3, padding=1),
            nn.Flatten(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.layers(X)
        return output


class BaseDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            layers.DecoderBlock(72, 64, 3, padding=1),
            layers.DecoderBlock(64, 48, 3, padding=1),
            layers.DecoderBlock(48, 32, 3, padding=1),
            layers.DecoderBlock(32, 16, 3, padding=1),
            layers.DecoderBlock(16, 8, 3, padding=1),
            layers.DecoderBlock(8, 4, 3, padding=1),
            layers.DecoderBlock(4, 1, 3, activation=nn.Sigmoid(), upsample_size=2, padding=1, batch_norm=False),
        )

    def forward(self, X: torch.Tensor):
        output = self.layers(X)
        return output


class TransposeDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            layers.TransposeDecoderBlock(
                256, 128, 5, stride=4, padding=1, output_padding=3, dropout=dropout
            ),
            layers.TransposeDecoderBlock(
                128, 64, 2, stride=4, padding=1, output_padding=2, dropout=dropout
            ),
            layers.TransposeDecoderBlock(
                64, 32, 4, stride=3, padding=1, output_padding=2, dropout=dropout
            ),
            layers.TransposeDecoderBlock(
                32, 16, 3, stride=3, padding=1, output_padding=1, dropout=dropout
            ),
            layers.TransposeDecoderBlock(
                16, 8, 2, stride=3, padding=0, output_padding=1, dropout=dropout
            ),
            layers.TransposeDecoderBlock(
                8, 1, 2, stride=2, padding=0, batch_norm=False, activation=nn.Sigmoid()
            ),
        )

    def forward(self, X: torch.Tensor):
        return self.model(X)


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.apply(initialize_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        return self.decoder(z).squeeze()

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
        return {"validation_loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs: List[dict]) -> dict:
        avg_loss = torch.stack([x["validation_loss"] for x in outputs]).mean()
        tensorboard_logs = {"validation_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}


class UNetAutoEncoder(AutoEncoder):
    def __init__(self, lr=1e-3, skim=True):
        super().__init__(encoder=None, decoder=None, lr=lr)
        if skim:
            self.model = SkimUNet(1, 1)
        else:
            self.model = UNet(1, 1)
        self.apply(initialize_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.ndim == 4
        return self.model(X)


class VAE(AutoEncoder):
    def __init__(
        self,
        encoder,
        decoder,
        beta=4,
        encoding_imgsize=8,
        encoding_filters=72,
        latent_dim=64,
        lr=0.001,
    ):
        super().__init__(encoder, decoder, lr=lr)
        self.encoding_imgsize = encoding_imgsize
        self.encoding_dim = encoding_imgsize ** 2 * encoding_filters
        self.fc_mu = nn.Linear(self.encoding_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoding_dim, latent_dim),
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, self.encoding_dim),
            nn.LeakyReLU(0.3, inplace=True)
            )
        self.beta = beta
        self.latent_dim = latent_dim
        self.encoding_filters = encoding_filters
        self.apply(initialize_weights)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return std * eps + mu

    def encode(self, X: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate an encoding

        Parameters
        ----------
        input : Tensor
            [description]

        Returns
        -------
        List[Tensor]
            [description]
        """
        # this generates shape (N x 4608) from flattening
        # the filter x H x W
        output = self.encoder(X)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(output)
        # log_var is forced to prefer negative (small) values
        log_var = -F.leaky_relu(self.fc_logvar(output), 0.5)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        output = self.decoder_input(z).view(
            batch_size, self.encoding_filters, self.encoding_imgsize, self.encoding_imgsize
        )
        # reshape the decoder input back into image dimensions
        output = self.decoder(output)
        return output

    def loss_function(
        self,
        pred_Y: torch.Tensor,
        Y: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> dict:
        batch_size = Y.size(0)
        recon_loss = F.binary_cross_entropy(pred_Y, Y)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )
        # at the limit of beta=1, we have regular VAE
        loss = recon_loss + self.beta * batch_size * kld_loss
        return {"loss": loss, "reconstruction_loss": recon_loss, "kl_div": kld_loss}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z).squeeze()
        return output

    def training_step(self, batch, batch_idx) -> dict:
        X, Y = batch
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        pred_Y = self.decode(z)
        loss_dict = self.loss_function(pred_Y, Y.squeeze(), mu, log_var)
        return {"loss": loss_dict["loss"], "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        pred_Y = self.decode(z)
        loss_dict = self.loss_function(pred_Y, Y.squeeze(), mu, log_var)
        return {"validation_loss": loss_dict["loss"], "log": loss_dict}

    def validation_epoch_end(self, outputs: List[dict]) -> dict:
        avg_loss = torch.stack([x["validation_loss"] for x in outputs]).mean()
        tensorboard_logs = {"validation_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

