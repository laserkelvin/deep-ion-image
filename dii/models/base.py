from dii.models.resnet import resnet18_encoder
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
            layers.TransposeDecoderBlock(256, 128, 4, padding=1, stride=4, output_padding=3),
            layers.TransposeDecoderBlock(128, 64, 3, padding=2, stride=2, output_padding=1),
            layers.TransposeDecoderBlock(64, 48, 4, padding=1, stride=4, output_padding=2),
            layers.TransposeDecoderBlock(48, 24, 8, padding=5, stride=4),
            layers.TransposeDecoderBlock(24, 8, 4, padding=3, stride=2, output_padding=1),
            layers.TransposeDecoderBlock(8, 1, 4, activation=nn.Sigmoid(), batch_norm=False, stride=1),
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
            layers.DecoderBlock(8, 1, 3, activation=nn.Sigmoid(), batch_norm=False, upsample_size=1, padding=1),
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
        self.metric = nn.BCELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encoder(X)
        return self.decoder(z).squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        X, Y, _ = batch
        pred_Y = self.forward(X).squeeze()
        loss = self.metric(pred_Y.squeeze(), Y.squeeze())
        self.logger.experiment.log({"training_recon": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, _ = batch
        pred_Y = self.forward(X)
        loss = self.metric(pred_Y.squeeze(), Y.squeeze())
        self.logger.experiment.log({"validation_recon": loss})
        input_image = X[0].cpu().detach()
        target_image = Y[0].cpu().detach()
        pred_image = pred_Y[0].cpu().detach()
        return (loss, input_image, target_image, pred_image)

    def validation_epoch_end(self, outputs):
        _, input_image, target_image, pred_image = outputs[-1]
        image_size = input_image.size(-1)
        compressed = torch.cat([input_image.view(1, 1, image_size, image_size),
            target_image.view(1, 1, image_size, image_size),
            pred_image.view(1, 1, image_size, image_size)], dim=0
            )
        grid = torchvision.utils.make_grid(compressed)
        self.logger.experiment.log(
                {"alacarte": [wandb.Image(grid, caption="Input/Target/Predicted")]})


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
    def __init__(self, n_segs: int = 9, lr: float = 1e-3, bilinear: bool = True, **kwargs):
        super().__init__(encoder=None, decoder=None, lr=lr, **kwargs)
        self.encoder = UnetEncoder(1, bilinear)
        self.decoder = UnetDecoder(1, n_segs, bilinear=bilinear)
        self.apply(initialize_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(X)
        output, mask = self.decoder(encoding)
        return (output, mask)

    def step(self, batch, batch_idx):
        X, Y, mask = batch
        encoding = self.encoder(X)
        pred_Y, pred_mask = self.decoder(encoding)
        
        # calculate losses
        recon_loss = self.metric(pred_Y, Y)
        seg_loss = F.cross_entropy(pred_mask, mask)
        loss = recon_loss + seg_loss
        # get some images
        images = list()
        for tensor in [X, Y, pred_Y, mask, pred_mask]:
            images.append(
                tensor[0].cpu().detach()
            )
        logs = {
            "recon_loss": recon_loss,
            "seg_loss": seg_loss,
            "loss": loss,
            "samples": images
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items() if "samples" not in k}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items() if "samples" not in k})
        return (loss, logs)
    
    def validation_epoch_end(self, outputs):
        loss, logs = outputs[-1]
        images = logs.get("samples")
        x, y, pred_y, mask, pred_mask = images
        # just get the classes
        pred_mask = pred_mask.argmax(0).numpy()
        mask = mask.numpy()
        self.logger.experiment.log(
            {
                "target": wandb.Image(y, masks={
                    "ground_truth": {"mask_data": mask},
                    "prediction": {"mask_data": pred_mask}
                    }),
                "predicted": wandb.Image(pred_y, masks={
                    "ground_truth": {"mask_data": mask},
                    "prediction": {"mask_data": pred_mask}
                    }),
            }
        )

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
        self.metric = nn.BCELoss()
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


class PLVAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE.from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE.from_pretrained('stl10-resnet18')
    """
    def __init__(
        self,
        input_height: int,
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        
        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss
        
        # do some image logging as well
        # img_size = x.size(-1)
        # input_img = x[0].cpu().detach().view(1, img_size, img_size)
        # pred_img = x_hat[0].cpu().detach().view(1, img_size, img_size)
        # target_img = y[0].cpu().detach().view(1, img_size, img_size)
        # compressed = torch.cat([input_img, target_img, pred_img], dim=1
        #     )
        # grid = torchvision.utils.make_grid(compressed)
        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items() if "images" not in k}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items() if "images" not in k})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
