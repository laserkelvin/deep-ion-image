from typing import Tuple, Union, Iterable

import h5py
import torch
import numpy as np
from torch.utils import data
from torchvision.transforms import Compose, ToTensor

from dii.pipeline.transforms import central_pipeline, projection_pipeline


class H5Dataset(data.Dataset):
    def __init__(
        self,
        path: str,
        key: str,
        transform: Union[None, Iterable, "Compose"] = None,
        target_transform: Union[None, Iterable, "Compose"] = None,
    ):
        """
        Instantiate the H5Dataset object, with the necessary arguments `path` and `key`
        corresponding to the path to the HDF5 file and the key referring to the dataset
        of interest.

        An optional argument is `transform`, which sets up a pipeline that sequentially
        transforms the data before giving it up.

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        key : str
            Name of the target dataset within the HDF5 file
        transform : Union[None, Iterable,, optional
            A defined pipeline for sequential transforms. This arg expects either an
            iterable, which will then be `Compose`'d, or just a straight `Compose`
            object. By default None, which does nothing.
        """
        self.file_path = path
        self.key = key
        self.dataset = None
        if not transform:
            self.transform = projection_pipeline
        else:
            if type(transform) == list or type(transform) == tuple:
                self.transform = Compose(transform)
            else:
                self.transform = transform
        if not transform:
            self.target_transform = central_pipeline
        else:
            if type(target_transform) == list or type(target_transform) == tuple:
                self.target_transform = Compose(target_transform)
            else:
                self.target_transform = target_transform
        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = len(file[self.key])

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.key]
        X = np.array(self.dataset[index])
        # if we have a transform pipeline, run it
        if self.transform:
            return self.transform(X)
        X_max = X.max()
        return X / X_max

    def __len__(self):
        return self.dataset_len


class CompositeH5Dataset(H5Dataset):
    def __init__(
        self,
        path: str,
        key: str,
        transform: Union[None, Iterable, "Compose"] = None,
        target_transform: Union[None, Iterable, "Compose"] = None,
        scale: float = 2.0,
        seed=None,
        indices=None,
    ):
        """
        Inheriting from `H5Dataset`, this version is purely stochastic by generating
        new composite images every time an item is retrieved; you will never get the
        image you ask for with this class!!

        To generate the composite images, an exponential distribution is sampled to
        obtain the number of images to compose: the idea is that many images overlapping
        should not be overwhelming, since in many cases we're only dealing with single
        distributions.

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        key : str
            Name of the target dataset within the HDF5 file
        transform : Union[None, Iterable,, optional
            A defined pipeline for sequential transforms. This arg expects either an
            iterable, which will then be `Compose`'d, or just a straight `Compose`
            object. By default None, which does nothing.
        scale : float, optional
            The length scale for the exponential, by default 2.
        seed : [type], optional
            Random seed to use for the image generation, by default None
        """
        super().__init__(path, key, transform)
        self.seed = seed
        self.scale = scale
        self.indices = None
        # prescribed indices specify which images correspond to training
        # testing and validation, etc
        if indices is None:
            self.indices = np.arange(len(self))
        else:
            self.indices = indices

    def __len__(self):
        if self.indices is None:
            return self.dataset_len
        else:
            return self.indices.size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate an ion image. The `index` arg is never actually used, because
        images are generated randomly anyway.

        A 2-tuple is returned, both corresponding to the ion images. In the
        case of a `transform` pipeline being defined, the first element is
        the pipeline result, whereas the second element is the original image.

        Parameters
        ----------
        index : [type]
            Not used in this method, but kept for consistency.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
        """
        # use the projection set as the "dataset"
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.key]
        self.target_dataset = h5py.File(self.file_path, "r")["true"]
        # get the number of images to compose with, sampled from an exponential
        # decay distribution
        n_composites = int(np.random.exponential(self.scale) + 1)
        # choose the images randomly
        chosen = np.random.choice(self.indices, replace=False, size=n_composites)
        if n_composites != 1:
            chosen = sorted(chosen)
        # get the true central image and projections
        true = np.array(self.target_dataset[chosen])
        projection = np.array(self.dataset[chosen])
        # if we have multiple images, flatten to a single composite
        if true.ndim == 3:
            true = true.sum(axis=0)
        if projection.ndim == 3:
            projection = projection.sum(axis=0)
        # if we have a compose pipeline defined, run it
        if self.target_transform:
            true = self.target_transform(true)
        # Y is the central slice, whereas X is the projection, which is
        # appropriate for the direction we're going
        projection = self.transform(projection)
        # Normalize the image intensity to [0, 1]
        projection = projection / (projection.max() + 1e-9)
        true = true / (true.max() + 1e-9)
        return (projection, true)


class SelectiveComposite(H5Dataset):
    def __init__(self, path: str, key: str, transform=None, normalize=True):
        super().__init__(path, key, transform, normalize)

    def __getitem__(self, indices):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.key]
        indices = sorted(indices)
        X = self.dataset[indices].sum(axis=0).astype(np.float32)
        if self.transform:
            target = self.transform(X)
        else:
            target = X
        if self.normalize:
            np.divide(X, X.max(), out=X)
        # otherwise, just make a channel dimension
        X = torch.FloatTensor(X).unsqueeze(0)
        return (target, X)

    def __len__(self):
        if self.indices is None:
            return super().__len__(self)
        else:
            return len(self.indices)
