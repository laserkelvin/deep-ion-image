import abel
import numpy as np
import torch
from torch import nn
from torchvision import transforms as tf
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from skimage.util import random_noise


class AbelProjection(object):
    """
    Transform class that simply takes a NumPy 2D array and performs
    the forward Abel transform to generate the 3D projection to simulate
    an experimental image from a 2D image.
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return abel.Transform(X, direction="forward", method="hansenlaw").transform


class AddNoise(object):
    def __init__(self, mean: float = 0., var: float = 0.01):
        self.mean = mean
        self.var = var
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        # make sure to downcast the image, otherwise half and normal
        # precision will not work
        X = X / X.max()
        # flip a coin to determine Gaussian or Poisson noise
        if np.random.rand() >= 0.5:
            output = random_noise(X, "gaussian", clip=True, mean=self.mean, var=self.var)
        else:
            output = random_noise(X, "poisson", clip=True)
        return output.astype(np.float32)


class ProcessNumpyArray(object):
    """
    Transforms class that converts NumPy 2D arrays into PIL-ready formats.
    This is needed by the result of AbelProjection, which produces float32
    arrays. We first normalize and re-scale to a max of 255, followed by
    downcasting to uint8 expected by PIL.

    If the image is already a NumPy 2D np.uint8 array, we simply add a new
    dimension corresponding to a single "channel".

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __call__(self, X: np.ndarray):
        if X.dtype != np.uint8:
            X = X.astype(np.float32)
            X = X / X.max()
            X *= 255.0
            X = X.astype(np.uint8)
        return X[:, :, np.newaxis]


class ToNumpy(object):
    def __call__(self, X: Image) -> np.ndarray:
        return np.array(X, dtype=np.float32)


class BlurPIL(object):
    def __init__(self, max_blur=10.0):
        self.max_blur = max_blur

    def __call__(self, X: Image) -> Image:
        scale = np.random.uniform(1.0, self.max_blur)
        return X.filter(ImageFilter.GaussianBlur(scale))


class BlurImage(object):
    def __init__(self, max_blur=5.0):
        self.max_blur = max_blur

    def __call__(self, X: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(0.0, self.max_blur)
        return gaussian_filter(X, sigma=scale).astype(np.float32)


class Normalize(object):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        max_val = X.max()
        return X / (max_val + self.eps)

# this is a pipeline that has been tested and is known to provide the "right" kind
# of behaviour AFAIK
central_pipeline = tf.Compose(
    [
        # ProcessNumpyArray(),
        #tf.ToPILImage(),
        #tf.Resize((500,500)),
        #tf.RandomAffine(0.0, scale=(0.3, 1.0), resample=Image.BICUBIC),  # scale the image for randomness
        # BlurPIL(),
        tf.ToTensor(),
        Normalize(),
    ]
)

projection_pipeline = tf.Compose(
    [
        #tf.RandomAffine(
        #    0.0, translate=(0.05, 0.05), resample=Image.BICUBIC
        #),  # we move the 3D image around
        AddNoise(),
        tf.ToTensor(),
        Normalize(),
    ]
)

mini_forward_pipeline = tf.Compose(
    [
        ProcessNumpyArray(),
        tf.ToPILImage(),
        tf.Resize((256, 256)),
        tf.ToTensor(),
        Normalize()
    ]
)