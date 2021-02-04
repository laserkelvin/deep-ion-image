
from typing import Union

import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

def weight_visualization(model: nn.Module, n_rows: int = 8, padding: int = 1):
    for index, module in enumerate(model.modules()):
        mod_type = module.__class__.__name__
        if "Conv" in mod_type:
            weights = module.weight
            kernel_size = weights.size(-1)
            # flatten the layer into just kernels
            collapsed = weights.view(-1, kernel_size, kernel_size)
            n_filters = collapsed.size(0)
            # figure out how many axes we need
            rows = np.min((n_filters // n_rows + 1, 64))
            grid = make_grid(collapsed.unsqueeze(1), nrow=n_rows, normalize=True, padding=padding)
            fig = plt.figure(figsize=(n_rows, rows))
            plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
            fig.savefig(f"layer{index}_weights.png", dpi=150)


def get_input_gradients(model: nn.Module, X: torch.Tensor, Y: Union[None, torch.Tensor] = None):
    # make sure gradients are going to computed
    model.train()
    X = X.requires_grad_(True)
    pred_Y = model(X)
    if not Y:
        Y = torch.ones_like(X)
    # backprop w.r.t. inputs
    pred_Y.backward(gradient=Y)
    return X.grad.detach()