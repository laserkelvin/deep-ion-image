from typing import Dict, List, Union, Iterable
from argparse import ArgumentParser

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
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        nn.init.sparse_(m.weight, sparsity=0.05)
    else:
        pass


def common_hyperparameters(parent_parser):
    """
    Tucks the argparse stuff into the model.
    
    See https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html?highlight=argpars
    for details.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--in_channels', type=int, default=1, metavar='N',
                help='Number of input channels (default: 1)')
    parser.add_argument('--out_channels', type=int, default=1, metavar='N',
                        help='Number of output channels (default: 1)')
    parser.add_argument('--latent_dim', type=int, default=64, metavar='N',
                        help='Dimensionality of the latent vector (default: 64)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--n_workers', type=int, default=8, metavar='N',
                        help='Number of CPUs for dataloading (default: 9)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--train_seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--test_seed', type=int, default=1923, metavar='S',
                        help='random seed (default: 1923)')
    parser.add_argument('--activation', type=str, default="silu", metavar='GC',
                        help='Name of activation function (default: silu)')
    parser.add_argument('--dropout', type=float, default=0., metavar='GC',
                        help='Dropout probability (default: 0)')
    parser.add_argument('--weight_decay', type=float, default=0., metavar='GC',
                        help='L2 regularization (default: 0)')
    parser.add_argument('--pretraining', type=bool, default=False, metavar='GC',
                        help='Whether to use composite images or not. This is helpful for checking if learning is happening at all (default: False)')
    parser.add_argument('--h5_path', type=str, default="../data/raw/128_ion_images.h5", metavar='GC',
                        help='HDF5 ion image dataset path')
    return parser


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
        model.extend([nn.Flatten(), nn.Linear(128 * 3 * 3, latent_dim)])
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
                        batch_norm=False,
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
        self.metric = nn.BCELoss()
        # do non-standard weight initialization
        self.apply(initialize_weights)
        # save __init__ args to hparams attribute
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = common_hyperparameters(parent_parser)
        return parser

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def _reconstruction_loss(self, pred_Y: torch.Tensor, Y: torch.Tensor):
        # compute the pixelwise loss
        pixelwise_loss = self.metric(pred_Y, Y)
        loss = pixelwise_loss
        return loss

    def step(self, batch, batch_idx):
        # for pretraining, we want the model to learn the noisefree stuffs
        if self.hparams.pretraining:
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

    def train_dataloader(self) -> DataLoader:
        if not self.hparams.pretraining:
            dataset_type = datautils.CompositeH5Dataset
            target = "projection"
            transform = transforms.projection_pipeline
        else:
            dataset_type = datautils.H5Dataset
            target = "true"
            transform = transforms.central_pipeline
        dataset = dataset_type(
            self.hparams.h5_path,
            target,
            indices=self.hparams.train_indices,
            seed=self.hparams.train_seed,
            transform=transform,
        )
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers)
        return loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if not self.hparams.pretraining:
            dataset_type = datautils.CompositeH5Dataset
            target = "projection"
            transform = transforms.projection_pipeline
        else:
            dataset_type = datautils.H5Dataset
            target = "true"
            transform = transforms.central_pipeline
        dataset = dataset_type(
            self.hparams.h5_path,
            target,
            indices=self.hparams.test_indices,
            seed=self.hparams.test_seed,
            transform=transform,
        )
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers)
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
        self.apply(initialize_weights)
        # in the event KLdiv goes to NaN, make sure weights are small
        # nn.init.uniform_(self.fc_logvar.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.fc_logvar.bias, -1e-3, 1e-3)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = common_hyperparameters(parent_parser)
        parser.add_argument('--beta', type=float, default=1., metavar='GC',
                    help='Beta coefficient for prior regularization (default: 1)')
        return parser

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
        # calculate the prior regularization term
        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.hparams.beta
        loss = recon_loss + kl
        logs = {"recon_loss": recon_loss, "kl": kl, "loss": loss}
        return loss, logs

    def step(self, batch, batch_idx):
        if not self.hparams.pretraining:
            X, Y, _, unsplit = batch
        # don't use the noisy images for pretraining
        else:
            X = batch
            Y = torch.clone(X)
            unsplit = None
        z, pred_Y, p, q = self._run_step(X)
        if self.hparams.split_true and not self.hparams.pretraining:
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
        weight_decay: float = 0.0,
        split_true: bool = False,
        activation: str = "prelu",
        dropout: float = 0.0,
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
        self.autoencoder = AutoEncoder(
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
        # set up the discriminator
        self.discriminator = nn.Sequential(
            Encoder(in_channels, latent_dim=8, activation=activation, dropout=dropout),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.discrim_metric = nn.BCELoss()
        # initialize weights for the discriminator
        self.discriminator.apply(initialize_weights)
        self.save_hyperparameters()

    def configure_optimizers(self):
        lr, weight_decay = self.hparams.lr, self.hparams.weight_decay
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, weight_decay=weight_decay
        )
        opt_gen = torch.optim.Adam(
            self.autoencoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        return [opt_disc, opt_gen], []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(X)

    def _disc_loss(self, X: torch.Tensor, Y: torch.Tensor):
        # training on real data first
        real_pred = self.discriminator(Y)
        real_ones = torch.ones_like(real_pred)
        real_loss = self.discrim_metric(real_pred, real_ones)

        # now run the VAE model to generate an image
        samples = self.autoencoder(X)
        fake_pred = self.discriminator(samples)
        fake_zeros = torch.zeros_like(fake_pred)
        fake_loss = self.discrim_metric(fake_pred, fake_zeros)

        disc_loss = fake_loss + real_loss
        logs = {
            "discriminator/fake": fake_loss,
            "discriminator/real": real_loss,
            "discriminator/combined": disc_loss,
        }
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
        logs = {
            "generator/fool": gen_loss,
            "generator/recon": recon_loss,
            "generator/combined": loss,
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
        pred_Y = self.autoencoder(X)
        images = list()
        index = np.random.randint(0, X.size(0))
        for tensor in [X, Y, pred_Y]:
            images.append(tensor[index].detach().cpu())
        logs["samples"] = images
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


class VAEGAN(AEGAN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta: float = 4.0,
        latent_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        split_true: bool = False,
        activation: str = "prelu",
        dropout: float = 0.0,
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
            lr=lr,
            weight_decay=weight_decay,
            split_true=split_true,
            activation=activation,
            dropout=dropout,
            train_seed=train_seed,
            test_seed=test_seed,
            n_workers=n_workers,
            batch_size=batch_size,
            h5_path=h5_path,
            train_indices=train_indices,
            test_indices=test_indices,
            **kwargs,
        )
        # swap out the autoencoder for a variational one
        del self.autoencoder
        self.autoencoder = VAE(
            in_channels,
            out_channels,
            beta=beta,
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
        self.save_hyperparameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(X)

    def _gen_loss(self, X: torch.Tensor, Y: torch.Tensor):
        z, pred_images, p, q = self.autoencoder._run_step(X)
        # this is modified to use the VAE loss instead of the straight
        # autoencoder loss
        vae_loss, logs = self.autoencoder._vae_loss(Y, pred_images, z, p, q)
        fake_pred = self.discriminator(pred_images)
        fake_ones = torch.ones_like(fake_pred)
        gen_loss = self.discrim_metric(fake_pred, fake_ones)
        loss = vae_loss + gen_loss
        # log the D(G(X)) loss - i.e. feedback on how the image is bad
        logs["generator/fool"] = gen_loss
        images = list()
        index = np.random.randint(0, X.size(0))
        for tensor in [X, Y, pred_images]:
            images.append(tensor[index].detach().cpu())
        logs["samples"] = images
        return loss, logs

    def validation_epoch_end(self, outputs):
        _, logs = outputs[-1]
        images = logs.get("samples")
        x, y, pred_y = images
        # make sure the samples are on the same device as the model
        X = x.repeat((16, 1, 1, 1)).to(self.device)
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


# this creates a mapping useful for argparse
valid_models = {
    "baseline": AutoEncoder,
    "vae": VAE,
    "aegan": AEGAN,
    "vaegan": VAEGAN
}
