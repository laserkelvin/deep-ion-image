import torch
from torchsummary import summary
from ..base import VAE, BaseEncoder, BaseDecoder


def test_vae():
    CUDA = torch.cuda.is_available()
    model = VAE(in_channels=1, out_channels=1, z_dim=32, latent_dim=32)
    device = "cuda" if CUDA else "cpu"
    #model.to(device)
    summary(model, (1, 128, 128), device=device)
