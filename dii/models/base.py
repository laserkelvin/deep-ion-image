from typing import Dict, List, Union, Iterable

import torch
import pytorch_lightning as pl
import wandb
import torchvision
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dii.models import layers
from dii.models.unet import UnetEncoder, UnetDecoder
from dii.pipeline import datautils, transforms
from dii.visualization.visualize import get_input_gradients
from pl_bolts.models.vision import PixelCNN


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # based on the original Kaiming paper, this helps deeper
        # convolutions not vanish
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Linear):
        nn.init.sparse_(m.weight, sparsity=0.05)
    else:
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
        model = nn.ModuleList()
        for index, out_channels in enumerate(sizes):
            if index == 0:
                model.append(
                    layers.ResidualBlock(
                        in_channels,
                        out_channels,
                        pool=2,
                        kernel_size=7,
                        use_1x1conv=False,
                        activation=chosen_activation,
                        dropout=dropout,
                    )
                    )
            else:
                model.append(
                    layers.ResidualBlock(
                        sizes[index - 1],
                        out_channels,
                        pool=2,
                        kernel_size=3,
                        use_1x1conv=True,
                        activation=chosen_activation,
                        dropout=dropout,
                    )
                )
        model.extend(
            [nn.Flatten(), nn.Linear(128 * 3 * 3, latent_dim)]
            )
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
        sizes = [128, 64, 32, 16, 8, 4, out_channels]
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
        model = nn.ModuleList()
        for index, out_channels in enumerate(sizes):
            if index == 0:
                pass
            # no activation for the last layer
            elif index == len(sizes) - 1:
                model.append(
                    layers.DecoderBlock(
                        sizes[index - 1],
                        out_channels,
                        kernel_size=7,
                        upsample_size=2,
                        activation=nn.Sigmoid,
                        batch_norm=False
                    )
                )
            else:
                model.append(
                    layers.DecoderBlock(
                        sizes[index - 1],
                        out_channels,
                        kernel_size=5,
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
        pretraining: bool = False,
        train_seed: int = 42,
        test_seed: int = 1923,
        n_workers: int = 8,
        batch_size: int = 64,
        h5_path: Union[None, str] = None,
        train_indices: Union[np.ndarray, Iterable, None] = None,
        test_indices: Union[np.ndarray, Iterable, None] = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, activation, dropout)
        self.decoder = Decoder(latent_dim, out_channels, activation, dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.split_true = split_true
        self.apply(initialize_weights)
        self.metric = nn.BCELoss()
        #self.metric = nn.MSELoss()
        self.pretraining = pretraining
        self.train_settings = {
            "train_seed": train_seed,
            "test_seed": test_seed,
            "n_workers": n_workers,
            "batch_size": batch_size,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }
        self.h5_path = h5_path
        self.save_hyperparameters() 

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def _reconstruction_loss(self, pred_Y: torch.Tensor, Y: torch.Tensor):
        # compute the pixelwise loss
        pixelwise_loss = self.metric(pred_Y, Y)
        loss = pixelwise_loss
        return loss

    def step(self, batch, batch_idx):
        # for pretraining, we want the model to learn the noisefree stuffs
        if self.pretraining:
            X = batch
            Y = torch.clone(X)
        else:
            # ignore the mask and the unsplit images
            X, Y, _, unsplit = batch
        pred_Y = self(X)
        if self.split_true and not self.pretraining:
            loss = self.metric(pred_Y, unsplit.permute(0, 2, 1, 3))
            collapsed = pred_Y.sum(1).unsqueeze(1)
            loss += self._reconstruction_loss(collapsed, Y)
            pred_Y = collapsed
        else:
            loss = self._reconstruction_loss(pred_Y, Y)
        # get some images
        images = list()
        index = np.random.randint(0, X.size(0))
        for tensor in [X, Y, pred_Y]:
            images.append(tensor[index].cpu().detach())
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

    def train_dataloader(self) -> DataLoader:
        train_indices = self.train_settings.get("train_indices")
        train_seed = self.train_settings.get("train_seed")
        batch_size = self.train_settings.get("batch_size")
        n_workers = self.train_settings.get("n_workers")
        if not self.pretraining:
            dataset_type = datautils.CompositeH5Dataset
            target = "projection"
            transform = transforms.projection_pipeline
        else:
            dataset_type = datautils.H5Dataset
            target = "true"
            transform = transforms.central_pipeline
        dataset = dataset_type(
            self.h5_path,
            target,
            indices=train_indices,
            seed=train_seed,
            transform=transform,
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)
        return loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_indices = self.train_settings.get("test_indices")
        test_seed = self.train_settings.get("test_seed")
        batch_size = self.train_settings.get("batch_size")
        n_workers = self.train_settings.get("n_workers")
        if not self.pretraining:
            dataset_type = datautils.CompositeH5Dataset
            target = "projection"
            transform = transforms.projection_pipeline
        else:
            dataset_type = datautils.H5Dataset
            target = "true"
            transform = transforms.central_pipeline
        dataset = dataset_type(
            self.h5_path,
            target,
            indices=test_indices,
            seed=test_seed,
            transform=transform,
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)
        return loader


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
        beta: float = 4.0,
        latent_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        split_true: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
        pretraining: bool = False,
        train_seed: int = 42,
        test_seed: int = 1923,
        n_workers: int = 8,
        batch_size: int = 64,
        h5_path: Union[None, str] = None,
        train_indices: Union[np.ndarray, Iterable, None] = None,
        test_indices: Union[np.ndarray, Iterable, None] = None,
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
            pretraining,
            train_seed,
            test_seed,
            n_workers,
            batch_size,
            h5_path,
            train_indices,
            test_indices,
            **kwargs,
        )
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.beta = beta
        self.latent_dim = latent_dim
        self.split_true = split_true
        # in the event KLdiv goes to NaN, make sure weights are small
        nn.init.uniform_(self.fc_logvar.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.fc_logvar.bias, -1e-3, 1e-3)
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

    def _vae_loss(self, Y, pred_Y, z, p, q):
        recon_loss = self._reconstruction_loss(pred_Y, Y)
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.beta
        loss = recon_loss + kl
        logs = {"recon_loss": recon_loss, "kl": kl, "loss": loss}
        return loss, logs

    def step(self, batch, batch_idx):
        if not self.pretraining:
            X, Y, _, unsplit = batch
        # don't use the noisy images for pretraining
        else:
            X = batch
            Y = torch.clone(X)
            unsplit = None
        z, pred_Y, p, q = self._run_step(X)
        if self.split_true and not self.pretraining:
            target = unsplit
        else:
            target = Y
        # compute losses and create a dictionary for logging
        loss, logs = self._vae_loss(target, pred_Y, z, p, q)
        images = list()
        index = np.random.randint(0, X.size(0))
        for tensor in [X, Y, pred_Y]:
            images.append(tensor[index].detach().cpu())
        # add sample images to logging
        logs["samples"] = images
        return loss, logs

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def validation_epoch_end(self, outputs):
        loss, logs = outputs[-1]
        images = logs.get("samples")
        x, y, pred_y = images
        X = x.repeat((16, 1, 1, 1)).to(self.device)
        with torch.no_grad():
            samples = self(X).cpu()
            sample_grid = torchvision.utils.make_grid(samples, nrow=4)
        self.logger.experiment.log(
            {
                "target": wandb.Image(y.float()),
                "predicted": wandb.Image(pred_y.float()),
                "input": wandb.Image(x.float()),
                "samples": wandb.Image(sample_grid),
            }
        )


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


class AEGAN(AutoEncoder):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.,
        split_true: bool = False,
        activation: str = "prelu",
        dropout: float = 0.,
        train_seed: int = 42,
        test_seed: int = 1923,
        n_workers: int = 8,
        batch_size: int = 64,
        h5_path: Union[None, str] = None,
        train_indices: Union[np.ndarray, Iterable, None] = None,
        test_indices: Union[np.ndarray, Iterable, None] = None,
        training: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            latent_dim=latent_dim,
            lr=lr,
            weight_decay=weight_decay,
            split_true=split_true,
            activation=activation,
            dropout=dropout,
            pretraining=False,
            train_seed=train_seed,
            test_seed=test_seed,
            n_workers=n_workers,
            batch_size=batch_size,
            h5_path=h5_path,
            train_indices=train_indices,
            test_indices=test_indices,
            **kwargs,
        )
        self.autoencoder = AutoEncoder(in_channels,
            out_channels,
            latent_dim=latent_dim,
            lr=lr,
            weight_decay=weight_decay,
            split_true=split_true,
            activation=activation,
            dropout=dropout,
            pretraining=False,
            train_seed=train_seed,
            test_seed=test_seed,
            n_workers=n_workers,
            batch_size=batch_size,
            h5_path=h5_path,
            train_indices=train_indices,
            test_indices=test_indices,
            **kwargs,)
        # for training, we'll use a discriminator too
        if training:
            self.discriminator = Encoder(in_channels, latent_dim=1, activation=activation, dropout=dropout)
            self.discrim_metric = nn.BCEWithLogitsLoss()
            # initialize weights
            self.encoder.apply(initialize_weights)
            self.decoder.apply(initialize_weights)
            self.discriminator.apply(initialize_weights)
        else:
            self.discriminator = None
        self.save_hyperparameters()

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr, weight_decay = self.hparams.lr, self.hparams.weight_decay
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=weight_decay)
        opt_gen = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
        return [opt_disc, opt_gen], []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(X)

    def _disc_loss(self, X: torch.Tensor, Y: torch.Tensor):
        # training on real data first
        real_pred = self.discriminator(Y)
        real_ones = torch.ones_like(real_pred)
        real_loss = self.discrim_metric(real_pred, real_ones)

        # now run the VAE model to generate an image
        samples = self(X)
        fake_pred = self.discriminator(samples)
        fake_zeros = torch.zeros_like(fake_pred)
        fake_loss = self.discrim_metric(fake_pred, fake_zeros)

        disc_loss = fake_loss + real_loss
        logs = {"discriminator/fake": fake_loss, "discriminator/real": real_loss, "discriminator/combined": disc_loss}
        return disc_loss, logs

    def _disc_step(self, X: torch.Tensor, Y: torch.Tensor):
        disc_loss = self._disc_loss(X, Y)
        return disc_loss

    def _gen_loss(self, X: torch.Tensor, Y: torch.Tensor):
        pred_Y = self.autoencoder(X)
        recon_loss = self.metric(pred_Y, Y)
        fake_pred = self.discriminator(pred_Y)
        fake_ones = torch.ones_like(fake_pred)
        # log the D(G(X)) loss - i.e. feedback on how the image is bad
        gen_loss = self.discrim_metric(fake_pred, fake_ones)
        loss = gen_loss + recon_loss
        images = list()
        for tensor in [X, Y, pred_Y]:
            images.append(tensor[0].cpu().detach())
        logs = {
            "generator/fool": gen_loss,
            "generator/recon": recon_loss,
            "generator/combined": loss,
            "samples": images
        }
        return loss, logs

    def _gen_step(self, X, Y):
        gen_loss = self._gen_loss(X, Y)
        return gen_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, Y, _, _ = batch
        loss, log = None, None
        # update the discriminator
        if optimizer_idx == 0 and self.current_epoch >= 5:
            loss, log = self._disc_step(X, Y)
        # update the generator
        if optimizer_idx == 1:
            loss, log = self._gen_step(X, Y)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, _, _ = batch
        gen_loss, logs = self._gen_loss(X, Y)
        return gen_loss, logs

    def validation_epoch_end(self, outputs):
        _, logs = outputs[-1]
        images = logs.get("samples")
        x, y, pred_y = images
        self.logger.experiment.log(
            {
                "target": wandb.Image(y.float()),
                "predicted": wandb.Image(pred_y.float()),
                "input": wandb.Image(x.float()),
            }
        )


class VAEGAN(AutoEncoder):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta: float = 4.,
        latent_dim: int = 128,
        z_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.,
        split_true: bool = False,
        activation: str = "prelu",
        dropout: float = 0.,
        train_seed: int = 42,
        test_seed: int = 1923,
        n_workers: int = 8,
        batch_size: int = 64,
        h5_path: Union[None, str] = None,
        train_indices: Union[np.ndarray, Iterable, None] = None,
        test_indices: Union[np.ndarray, Iterable, None] = None,
        training: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            beta=beta,
            latent_dim=latent_dim,
            z_dim=z_dim,
            lr=lr,
            weight_decay=weight_decay,
            split_true=split_true,
            activation=activation,
            dropout=dropout,
            pretraining=False,
            train_seed=train_seed,
            test_seed=test_seed,
            n_workers=n_workers,
            batch_size=batch_size,
            h5_path=h5_path,
            train_indices=train_indices,
            test_indices=test_indices,
            **kwargs,
        )
        self.autoencoder = VAE(in_channels,
            out_channels,
            beta=beta,
            latent_dim=latent_dim,
            z_dim=z_dim,
            lr=lr,
            weight_decay=weight_decay,
            split_true=split_true,
            activation=activation,
            dropout=dropout,
            pretraining=False,
            train_seed=train_seed,
            test_seed=test_seed,
            n_workers=n_workers,
            batch_size=batch_size,
            h5_path=h5_path,
            train_indices=train_indices,
            test_indices=test_indices,
            **kwargs,)
        # for training, we'll use a discriminator too
        if training:
            self.discriminator = Encoder(in_channels, latent_dim=1, activation=activation, dropout=dropout)
            self.discrim_metric = nn.BCEWithLogitsLoss()
            # initialize weights
            self.encoder.apply(self._weights_init)
            self.decoder.apply(self._weights_init)
            self.discriminator.apply(self._weights_init)
        else:
            self.discriminator = None
        self.save_hyperparameters()

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr, weight_decay = self.hparams.lr, self.hparams.weight_decay
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=weight_decay)
        opt_gen = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
        return [opt_disc, opt_gen], []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(X)

    def _disc_loss(self, X: torch.Tensor, Y: torch.Tensor):
        # training on real data first
        real_pred = self.discriminator(Y)
        real_ones = torch.ones_like(real_pred)
        real_loss = self.discrim_metric(real_pred, real_ones)

        # now run the VAE model to generate an image
        samples = self(X)
        fake_pred = self.discriminator(samples)
        fake_zeros = torch.zeros_like(fake_pred)
        fake_loss = self.discrim_metric(fake_pred, fake_zeros)

        disc_loss = fake_loss + real_loss
        logs = {"discriminator/fake": fake_loss, "discriminator/real": real_loss, "discriminator/combined": disc_loss}
        return disc_loss, logs

    def _disc_step(self, X: torch.Tensor, Y: torch.Tensor):
        disc_loss = self._disc_loss(X, Y)
        return disc_loss

    def _gen_loss(self, X: torch.Tensor, Y: torch.Tensor):
        z, pred_images, p, q = self.autoencoder._run_step(X)
        vae_loss, logs = self.autoencoder._vae_loss(Y, pred_images, z, p, q)
        fake_pred = self.discriminator(pred_images)
        fake_ones = torch.ones_like(fake_pred)
        gen_loss = self.discrim_metric(fake_pred, fake_ones)
        # log the D(G(X)) loss - i.e. feedback on how the image is bad
        logs["generator/fool"] = gen_loss
        images = list()
        for tensor in [X, Y, pred_images]:
            images.append(tensor[0].cpu().detach())
        logs["samples"] = images
        return gen_loss, logs

    def _gen_step(self, X, Y):
        gen_loss = self._gen_loss(X, Y)
        return gen_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, Y, _, _ = batch
        loss, log = None, None
        # update the discriminator
        if optimizer_idx == 0:
            loss, log = self._disc_step(X, Y)
        # update the generator
        if optimizer_idx == 1:
            loss, log = self._gen_step(X, Y)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, _, _ = batch
        gen_loss, logs = self._gen_loss(X, Y)
        return gen_loss, logs

    def validation_epoch_end(self, outputs):
        _, logs = outputs[-1]
        images = logs.get("samples")
        x, y, pred_y = images
        X = x.repeat((16, 1, 1, 1)).cuda()
        with torch.no_grad():
            samples = self.autoencoder(X).cpu()
            sample_grid = torchvision.utils.make_grid(samples, nrow=4)
        self.logger.experiment.log(
            {
                "target": wandb.Image(y.float()),
                "predicted": wandb.Image(pred_y.float()),
                "input": wandb.Image(x.float()),
                "samples": wandb.Image(sample_grid),
            }
        )
