from typing import Union, List, Iterable, Tuple, Callable, Dict

import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

from dii.pipeline.make_dataset import generate_image
from dii.pipeline.transforms import Normalize


def weight_visualization(model: nn.Module, n_rows: int = 8, padding: int = 1):
    for index, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            weights = module.weight
            kernel_size = weights.size(-1)
            # flatten the layer into just kernels
            collapsed = weights.view(-1, kernel_size, kernel_size)
            n_filters = collapsed.size(0)
            # figure out how many axes we need
            rows = np.min((n_filters // n_rows + 1, 64))
            grid = make_grid(
                collapsed.unsqueeze(1), nrow=n_rows, normalize=True, padding=padding
            )
            fig = plt.figure(figsize=(n_rows, rows))
            plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
            fig.savefig(f"layer{index}_weights.png", dpi=150)


def radial_profile(data: np.ndarray, center: Iterable[float]) -> np.ndarray:
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def show_torch_grid(X: Union[torch.Tensor, List[torch.Tensor]], ax=None, **kwargs):
    """
    Quick method for creating a matplotlib image grid from
    a tensor or a list of tensors. Wraps the `torchvision.utilts.make_grid`
    function, and `kwargs` are passed into this function.

    Parameters
    ----------
    X : Union[torch.Tensor, List[torch.Tensor]]
        Tensor(s) to make the image grid with
    ax : [type], optional
        Instance of a matplotlib `axis` object, by default None
    """
    # we don't have a channel dimension
    grid = make_grid(X, **kwargs).sum(dim=0)
    if not ax:
        return plt.imshow(grid)
    else:
        return ax.imshow(grid)


def half_half_image(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Fills one half of an image with the true and the other with
    the predicted distribution for side-by-side comparison. The
    image dimensions must be the same.

    Parameters
    ----------
    true : np.ndarray
        True/input image
    predicted : np.ndarray
        Reconstructed image, or really anything you like

    Returns
    -------
    np.ndarray
        Half/half image. The left corresponds to the input/true
        image, and the right side corresponds to the `predicted`.
    """
    assert true.shape == predicted.shape
    new_image = np.zeros_like(true)
    center = (true.shape[0] // 2)
    new_image[:,:center] = true[:,:center]
    new_image[:,center:] = predicted[:,center:]
    return new_image


def generate_multiple_rings(n_rings: int, img_size: int = 128, beta: Union[Iterable, float] = 0.) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an image with `n_rings` concentric, linearly spaced, isotropic rings.
    """
    try:
        # if we can't iterate over the beta values
        iter(beta)
    except TypeError:
        # if we're using one beta value, just copy it
        # `n_rings` times
        beta = [beta,] * n_rings
    center = img_size // 2
    # same as used for the image generation pipeline
    min_size, max_size = center * 0.1, center * 0.8
    # generate a number of concentric rings
    centers = np.linspace(min_size, max_size, n_rings)
    central_image = np.zeros((img_size, img_size), dtype=np.float32)
    projection = np.zeros_like(central_image)
    for center, b in zip(centers, beta):
        temp_central, temp_projection = generate_image(center, 1.5, b, img_size)
        central_image += (temp_central / temp_central.max())
        projection += (temp_projection / temp_projection.max())
    central_image /= central_image.max()
    projection /= projection.max()
    return (central_image, projection)


def add_noise(image: np.ndarray, rng: "Generator", var: float) -> Tuple[np.ndarray, float]:
    """
    Adds an amount of Gaussian and Poisson noise to the image.
    """
    gaussian_noise = rng.normal(0., var, size=image.shape)
    poisson_image = random_noise(image, "poisson", clip=True)
    result = (gaussian_noise + poisson_image).astype(np.float32)
    normalize = Normalize()
    result = normalize(result)
    image = normalize(image)
    snr = peak_signal_noise_ratio(image, result)
    return (result, snr)


class FeatureExtraction(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._features = {}

        for name, layer in self.model.named_modules():
            layer.__name__ = name
            self._features[name] = torch.empty(0)
            layer.register_forward_hook(self.save_outputs_hook(name))
    
    def save_outputs_hook(self, name: str) -> Callable:
        def fn(_, __, output):
            self._features[name] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features