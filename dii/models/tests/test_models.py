import torch
from ..base import VAE, BaseEncoder, BaseDecoder


def test_vae():
    images = torch.rand(16, 1, 500, 500)
    model = VAE(BaseEncoder(), BaseDecoder())
