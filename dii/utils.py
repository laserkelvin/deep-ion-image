import os

from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import numba


# Auxillary routines for random numerical checks

@numba.njit
def cart2r(x: float, y: float) -> float:
    r = np.sqrt(x**2 + y**2)
    return r


@numba.njit
def radial_integration(image: np.ndarray) -> np.ndarray:
    """
    Integrate the radial profile of a square image. This assumes that the image is
    centered, and returns a normalized distribution of "intensity" as a function
    of distance from the center.

    Parameters
    ----------
    image : np.ndarray
        Numpy 2D array, with dimensions N x N

    Returns
    -------
    np.ndarray
        NumPy 1D array corresponding to 
    """
    h, w = image.shape
    center = (h + w) // 4
    r_bins = np.zeros(center)
    for x in range(center):
        for y in range(center):
            r = np.int(cart2r(x, y))
            # for some reason, it wants to go out of bounds
            # sometimes
            if r < center:
                r_bins[r] += image[x+center, y+center]
    hist_max = r_bins.max()
    return r_bins / hist_max


# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='validation_loss',
    mode='min',
    prefix=''
)

