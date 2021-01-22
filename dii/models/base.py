from typing import Dict, List

import torch
import pytorch_lightning as pl
import wandb
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
            layers.ConvolutionBlock(
                1, 8, 3, padding=2, pool=nn.MaxPool2d(4), dropout=dropout
            ),  # 8 x 125 x 125
            layers.ConvolutionBlock(
                8, 48, 3, padding=2, pool=nn.MaxPool2d(2), dropout=dropout
            ),  # 48 x 63 x 63
            layers.ConvolutionBlock(
                48, 64, 3, padding=2, pool=nn.MaxPool2d(4), dropout=dropout
            ),  # 64 x 16 x 16
            layers.ConvolutionBlock(
                64, 128, 3, padding=2, pool=nn.MaxPool2d(2), dropout=dropout
            ),  # 128 x 9 x 9
            layers.ConvolutionBlock(
                128, 256, 3, padding=2, pool=nn.MaxPool2d(4), dropout=dropout
            ),  # 256 x 2 x 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.layers(X)
        return output


class BaseDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            layers.TransposeDecoderBlock(
                256, 128, 3, padding=1, stride=4, dropout=dropout
            ),  # 128 x 5 x 5
            layers.TransposeDecoderBlock(
                128, 64, 3, padding=0, stride=2, dropout=dropout
            ),  # 64 x 11 x 11
            layers.TransposeDecoderBlock(
                64, 48, 2, padding=2, stride=2, dropout=dropout
            ),  # 48 x 18 x 18
            layers.TransposeDecoderBlock(
                48, 24, 3, padding=3, stride=4, dropout=dropout
            ),  # 24 x 65 x 65
            layers.TransposeDecoderBlock(
                24, 8, 3, padding=2, stride=2, dropout=dropout
            ),  # 8 x 127 x 127
            layers.TransposeDecoderBlock(
                8,
                1,
                3,
                activation=nn.Sigmoid(),
                padding=4,
                output_padding=1,
                batch_norm=False,
                stride=4,
            ),  # 1 x 500 x 500
        )

    def forward(self, X: torch.Tensor):
        output = self.layers(X)
        return output


class UpsampleDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            layers.DecoderBlock(256, 128, 3, padding=1, upsample_size=4, dropout=dropout),
            layers.DecoderBlock(128, 64, 3, padding=1, dropout=dropout),
            layers.DecoderBlock(64, 48, 3, padding=1, upsample_size=4, dropout=dropout),
            layers.DecoderBlock(48, 24, 5, padding=0, dropout=dropout),
            layers.DecoderBlock(24, 8, 3, padding=2, dropout=dropout),
            layers.DecoderBlock(8, 1, 3, activation=nn.Sigmoid(), batch_norm=False, padding=1),
        )

    def forward(self, X: torch.Tensor):
        output = self.layers(X)
        return output


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-3, weight_decay=0.0, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.apply(initialize_weights)
        self.metric = nn.MSELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        return self.decoder(z).squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        X, Y = batch
        pred_Y = self.forward(X).squeeze()
        loss = self.metric(pred_Y, Y)
        self.log("training_recon", loss_dict.get("loss"))
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        pred_Y = self.forward(X).squeeze()
        loss = self.metric(pred_Y, Y)
        self.log("validation_recon", loss_dict.get("loss"))
        tensorboard_logs = {"validation_loss": loss}
        return {"validation_loss": loss, "log": tensorboard_logs}



class UNetAutoEncoder(AutoEncoder):
    def __init__(self, lr=1e-3, skim=True, **kwargs):
        super().__init__(encoder=None, decoder=None, lr=lr, **kwargs)
        if skim:
            self.model = SkimUNet(1, 1)
        else:
            self.model = UNet(1, 1)
        self.apply(initialize_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
        **kwargs
    ):
        super().__init__(encoder, decoder, lr=lr, **kwargs)
        self.encoding_imgsize = encoding_imgsize
        self.encoding_dim = (encoding_imgsize ** 2) * encoding_filters
        self.fc_mu = nn.Linear(self.encoding_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoding_dim, latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, self.encoding_dim), nn.LeakyReLU(0.3, inplace=True)
        )
        self.beta = beta
        self.latent_dim = latent_dim
        self.encoding_filters = encoding_filters
        self.metric = nn.MSELoss()
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
        # get batch size for flattening later
        batch_size = X.size(0)
        # run through encoder model, and flatten for linear layer
        output = self.encoder(X).view(batch_size, -1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(output)
        # log_var is forced to prefer negative (small) values
        log_var = -F.leaky_relu(self.fc_logvar(output), 0.5)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        output = self.decoder_input(z).view(
            batch_size,
            self.encoding_filters,
            self.encoding_imgsize,
            self.encoding_imgsize,
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
        recon_loss = self.metric(pred_Y, Y)
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
        loss_dict = self.loss_function(pred_Y.squeeze(), Y.squeeze(), mu, log_var)
        self.log("training_total", loss_dict.get("loss"))
        self.log("training_recon", loss_dict.get("reconstruction_loss"))
        self.log("training_kl", loss_dict.get("kl_div"))
        return {"loss": loss_dict["loss"]}

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        pred_Y = self.decode(z)
        loss_dict = self.loss_function(pred_Y.squeeze(), Y.squeeze(), mu, log_var)
        self.log("validation_total", loss_dict.get("loss"))
        self.log("validation_recon", loss_dict.get("reconstruction_loss"))
        self.log("validation_kl", loss_dict.get("kl_div"))
        # sample images for visual checking
        input_image = X[0].cpu().detach()
        target_image = Y[0].cpu().detach()
        pred_image = pred_Y[0].cpu().detach()
        self.logger.experiment.log(
            {
                "input_image": [wandb.Image(input_image, caption="Input")],
                "target_image": [wandb.Image(target_image, caption="Target")],
                "pred_image": [wandb.Image(pred_image, caption="Prediction")],
            }
        )
        return {"validation_loss": loss_dict["loss"]}
