import abel
import numpy as np
from torch import nn
from torchvision import transforms as tf
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter


class AbelProjection(object):
    """
    Transform class that simply takes a NumPy 2D array and performs
    the forward Abel transform to generate the 3D projection to simulate
    an experimental image from a 2D image.
    """

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return abel.Transform(X, direction="forward", method="hansenlaw").transform


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
            X_max = X.max()
            X /= X_max
            X *= 255
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
    def __init__(self, max_blur=5.):
        self.max_blur = max_blur
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(0., self.max_blur)
        return gaussian_filter(X, sigma=scale).astype(np.float32)


# this is a pipeline that has been tested and is known to provide the "right" kind
# of behaviour AFAIK
central_pipeline = tf.Compose(
    [
        ProcessNumpyArray(),
        tf.ToPILImage(),
        tf.RandomAffine(0.0, scale=(0.3, 1.0), resample=Image.BICUBIC),  # scale the image for randomness
        BlurPIL(),
        ToNumpy(),
    ]
)

projection_pipeline = tf.Compose(
    [
        AbelProjection(),  # Perform Abel transform to get 3D distribution
        BlurImage(),
        tf.ToPILImage(),
        BlurPIL(),         # Gaussian blur to remove Abel projection artifacts
        tf.RandomAffine(0.0, translate=(0.05, 0.05), resample=Image.BICUBIC),    # we move the 3D image around
        tf.ToTensor(),
        nn.Dropout(0.2),   # This adds some noise to the 3D image
    ]
)
