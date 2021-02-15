from typing import Union, List, Iterable

import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


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
    grid = make_grid(X, **kwargs)
    if not ax:
        return plt.imshow(grid.permute(1, 2, 0))
    else:
        return ax.imshow(grid.permute(1, 2, 0))


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