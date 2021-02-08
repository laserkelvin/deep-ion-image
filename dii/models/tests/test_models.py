import torch
from torchsummary import summary
from ..base import VAE, AutoEncoder, PixelVAE


def _run_stack(model: torch.nn.Module, in_channels: int = 1, img_size: int = 128, cuda: bool = True):
    # uses torchsummary to run through the model and ensures that the
    # computation runs without a hitch
    devices = ["cpu"]
    if torch.cuda.is_available() and cuda:
        devices.append("cuda")
    for device in devices:
        model.to(device)
        summary(model, (in_channels, img_size, img_size), device=device)


def _check_shape(model: torch.nn.Module, in_channels: int = 1, img_size: int = 128):
    # this checks that the input and output image sizes are the same
    X = torch.rand(64, in_channels, img_size, img_size, device=model.device)
    assert X.shape == model(X).shape
    assert model(X).shape == (64, in_channels, img_size, img_size)


def test_autoencoder():
    # first test default sizes
    model = AutoEncoder(in_channels=1, out_channels=1)
    _run_stack(model)
    # now test non-default dimensions to make sure the encoder-decoder
    # stack is talking properly
    model = AutoEncoder(in_channels=1, out_channels=1, latent_dim=256)
    _run_stack(model)
    # now see if some other parameters break stuff
    model = AutoEncoder(in_channels=1, out_channels=1, latent_dim=32, drop_size=3, dropout=0.2)
    _run_stack(model)
    # now check the output shape
    _check_shape(model)


def test_vae():
    # first test default sizes
    model = VAE(in_channels=1, out_channels=1)
    _run_stack(model)
    # now test non-default dimensions to make sure the encoder-decoder
    # stack is talking properly
    model = VAE(in_channels=1, out_channels=1, latent_dim=32)
    _run_stack(model)
    # now check the output shape
    _check_shape(model)


def test_pixelvae():
    # first test default sizes
    model = PixelVAE(in_channels=1, out_channels=1)
    _run_stack(model, cuda=False)
    # see other settings
    model = PixelVAE(in_channels=1, out_channels=1, pixel_hidden=64, pixel_layers=5)
    _run_stack(model, cuda=False)
    # now check the output shape
    _check_shape(model)