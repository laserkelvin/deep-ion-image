import abel
import numpy as np
from torchvision import transforms as tf


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

# this is a pipeline that has been tested and is known to provide the "right" kind
# of behaviour AFAIK
default_pipeline = transforms.Compose(
    [
        AbelProjection(),
        ProcessNumpyArray(),
        tf.ToPILImage(),
        # this performs random translation and scaling to the image.
        tf.RandomAffine(0.0, translate=(0.05, 0.05), scale=(0.3, 1.0)),
        tf.ToTensor(),
    ]
)

