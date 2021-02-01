from typing import Dict, List

import torch
import pytorch_lightning as pl
import wandb
import torchvision
from torch import nn
from torch.nn import functional as F

from dii.models import layers
from dii.models.unet import SkimUNet, UnetEncoder, UnetDecoder
from dii.models.resnet import resnet18_encoder, resnet18_decoder
from pl_bolts.models.vision import PixelCNN


def initialize_weights(m):
    for name, parameter in m.named_parameters():
        if "norm" not in name:
            try:
                nn.init.xavier_uniform_(parameter)
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
                256, 128, 4, padding=1, stride=4, output_padding=3
            ),
            layers.TransposeDecoderBlock(
                128, 64, 3, padding=2, stride=2, output_padding=1
            ),
            layers.TransposeDecoderBlock(
                64, 48, 4, padding=1, stride=4, output_padding=2
            ),
            layers.TransposeDecoderBlock(48, 24, 8, padding=5, stride=4),
            layers.TransposeDecoderBlock(
                24, 8, 4, padding=3, stride=2, output_padding=1
            ),
            layers.TransposeDecoderBlock(
                8, 1, 4, activation=nn.Sigmoid(), batch_norm=False, stride=1
            ),
        )

    def forward(self, X: torch.Tensor):
        output = self.layers(X)
        return output


class UpsampleDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            layers.DecoderBlock(256, 128, 3, padding=2, upsample_size=4, reflection=2),
            layers.DecoderBlock(128, 64, 3, padding=1, reflection=1),
            layers.DecoderBlock(64, 48, 5, padding=2, upsample_size=2),
            layers.DecoderBlock(48, 24, 3, reflection=1),
            layers.DecoderBlock(24, 12, 3, padding=1, reflection=1),
            layers.DecoderBlock(12, 8, 5, upsample_size=2),
            layers.DecoderBlock(
                8,
                1,
                3,
                activation=nn.Sigmoid(),
                batch_norm=False,
                upsample_size=1,
                padding=1,
            ),
        )

    def forward(self, X: torch.Tensor):
        output = self.layers(X)
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        sizes = [
            8,
            16,
            32,
            64,
            128,
        ]
        activation_maps = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "prelu": nn.PReLU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation not in activation_maps:
            activation = "relu"
        chosen_activation = activation_maps.get(activation, "relu")
        for index, out_channels in enumerate(sizes):
            if index == 0:
                model = [
                    layers.ResidualBlock(
                        in_channels,
                        out_channels,
                        pool=2,
                        use_1x1conv=True,
                        activation=chosen_activation,
                        dropout=dropout,
                    )
                ]
            else:
                model.append(
                    layers.ResidualBlock(
                        sizes[index - 1],
                        out_channels,
                        pool=2,
                        use_1x1conv=True,
                        activation=chosen_activation,
                        dropout=dropout,
                    )
                )
        model.extend([nn.Flatten(), nn.Linear(128 * 4 * 4, latent_dim)])
        self.model = nn.Sequential(*model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        sizes = [128, 64, 32, 16, 8, out_channels]
        self.fc = nn.Linear(latent_dim, sizes[0] * 4 * 4)
        # get a mapping of valid activation functions
        activation_maps = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "prelu": nn.PReLU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation not in activation_maps:
            activation = "relu"
        chosen_activation = activation_maps.get(activation, "relu")
        model = list()
        for index, out_channels in enumerate(sizes):
            if index == 0:
                pass
            # no activation for the last layer
            elif index == len(sizes):
                model.append(
                    layers.DecoderBlock(
                        sizes[index - 1],
                        out_channels,
                        3,
                        upsample_size=2,
                        activation=None,
                        dropout=dropout,
                    )
                )
            else:
                model.append(
                    layers.DecoderBlock(
                        sizes[index - 1],
                        out_channels,
                        3,
                        upsample_size=2,
                        activation=chosen_activation,
                        dropout=dropout,
                    )
                )
        self.model = nn.Sequential(*model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.fc(X).view(-1, 128, 4, 4)
        return self.model(output)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int = 128,
        lr=1e-3,
        weight_decay=0.0,
        split_true: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, activation, dropout)
        self.decoder = Decoder(latent_dim, out_channels, activation, dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.split_true = split_true
        self.apply(initialize_weights)
        self.metric = nn.BCEWithLogitsLoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def step(self, batch, batch_idx):
        # ignore the mask and the unsplit images
        X, Y, _, unsplit = batch
        pred_Y = self(X)
        if self.split_true:
            loss = self.metric(pred_Y, unsplit.permute(0, 2, 1, 3))
            collapsed = pred_Y.sum(1).unsqueeze(1)
            loss += self.metric(collapsed, Y)
            pred_Y = collapsed
        else:
            loss = self.metric(pred_Y, Y)
        # get some images
        images = list()
        for tensor in [X, Y, pred_Y]:
            images.append(tensor[0].cpu().detach())
        logs = {"recon_loss": loss, "samples": images}
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items() if "samples" not in k},
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"val_{k}": v for k, v in logs.items() if "samples" not in k},
            on_step=False,
            on_epoch=True,
        )
        return (loss, logs)

    def validation_epoch_end(self, outputs):
        loss, logs = outputs[-1]
        images = logs.get("samples")
        x, y, pred_y = images
        self.logger.experiment.log(
            {
                "target": wandb.Image(y.float()),
                "predicted": wandb.Image(pred_y.float()),
                "input": wandb.Image(x.float()),
            }
        )


class UNetAutoEncoder(AutoEncoder):
    def __init__(self, lr=1e-3, bilinear=True, **kwargs):
        super().__init__(encoder=None, decoder=None, lr=lr, **kwargs)
        self.encoder = UnetEncoder(1, bilinear)
        self.decoder = UnetDecoder(1, 1, bilinear)
        self.apply(initialize_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(X)
        output, mask = self.decoder(encoding)
        return output


class UNetSegAE(AutoEncoder):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_segs: int = 9,
        start_features: float = 16,
        lr: float = 1e-3,
        bilinear: bool = True,
        activation=None,
        **kwargs,
    ):
        super().__init__(encoder=None, decoder=None, lr=lr, **kwargs)
        self.encoder = UnetEncoder(in_channels, start_features, bilinear, activation)
        self.decoder = UnetDecoder(
            out_channels, num_segs, start_features, bilinear, activation
        )
        self.apply(initialize_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(X)
        output, mask = self.decoder(encoding)
        return (output, mask)

    def step(self, batch, batch_idx):
        X, Y, mask, _ = batch
        encoding = self.encoder(X)
        pred_Y, pred_mask = self.decoder(encoding)

        # calculate losses
        recon_loss = self.metric(pred_Y, Y)
        seg_loss = F.cross_entropy(pred_mask, mask)
        loss = recon_loss + seg_loss
        # get some images
        images = list()
        for tensor in [X, Y, pred_Y, mask, pred_mask]:
            images.append(tensor[0].cpu().detach())
        logs = {
            "recon_loss": recon_loss,
            "seg_loss": seg_loss,
            "loss": loss,
            "samples": images,
        }
        return loss, logs

    def validation_epoch_end(self, outputs):
        loss, logs = outputs[-1]
        images = logs.get("samples")
        x, y, pred_y, mask, pred_mask = images
        # just get the classes
        pred_mask = pred_mask.argmax(0).numpy()
        mask = mask.numpy()
        self.logger.experiment.log(
            {
                "input": wandb.Image(
                    x.float(),
                    masks={
                        "ground_truth": {"mask_data": mask},
                        "prediction": {"mask_data": pred_mask},
                    },
                ),
                "target": wandb.Image(
                    y.float(),
                    masks={
                        "ground_truth": {"mask_data": mask},
                        "prediction": {"mask_data": pred_mask},
                    },
                ),
                "predicted": wandb.Image(
                    pred_y.float(),
                    masks={
                        "ground_truth": {"mask_data": mask},
                        "prediction": {"mask_data": pred_mask},
                    },
                ),
            }
        )


class VAE(AutoEncoder):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta=4,
        latent_dim=64,
        z_dim=64,
        lr=0.001,
        weight_decay: float = 0.,
        split_true: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            latent_dim,
            lr,
            weight_decay,
            split_true,
            activation,
            dropout,
            **kwargs,
        )
        self.fc_mu = nn.Linear(latent_dim, z_dim)
        self.fc_logvar = nn.Linear(latent_dim, z_dim)
        # in the event KLdiv goes to NaN, make sure weights are small
        # nn.init.uniform_(self.fc_logvar.weight, -1e-3, 1e-3)
        # nn.init.uniform_(self.fc_logvar.bias, -1e-3, 1e-3)
        self.beta = beta
        self.latent_dim = latent_dim
        self.split_true = split_true
        self.apply(initialize_weights)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        X, Y, _, unsplit = batch
        z, pred_Y, p, q = self._run_step(X)
        if self.split_true:
            recon_loss = self.metric(pred_Y, unsplit)
        else:
            recon_loss = self.metric(pred_Y, Y)

        images = list()
        for tensor in [X, Y, pred_Y]:
            images.append(tensor[0].cpu().detach())

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.beta

        loss = kl + recon_loss
        logs = {"recon_loss": recon_loss, "kl": kl, "loss": loss, "samples": images}
        return loss, logs

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)


class PixelCNNAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 256,
        num_blocks: int = 5,
        lr=1e-3,
        **kwargs,
    ):
        super().__init__(encoder=None, decoder=None, lr=lr, **kwargs)
        self.model = PixelCNN(input_channels, latent_dim, num_blocks)
        self.metric = nn.MSELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
