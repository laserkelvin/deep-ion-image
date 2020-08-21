from typing import Tuple, Union, Iterable

import h5py
import torch
from torch.utils import data
from torchvision.transforms import Compose


class H5Dataset(data.Dataset):
    def __init__(
        self, path: str, key: str, transform: Union[None, Iterable, "Compose"] = None
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
        if type(transform) == list or type(transform) == tuple:
            self.transform = Compose(transform)
        else:
            self.transform = transform
        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = len(file[self.key])

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.key]
        X = self.dataset[index]
        # if we have a transform pipeline, run it
        if self.transform:
            return self.transform(X)
        return X

    def __len__(self):
        return self.dataset_len


class CompositeH5Dataset(H5Dataset):
    def __init__(
        self,
        path: str,
        key: str,
        transform: Union[None, Iterable, "Compose"] = None,
        scale: float = 2.0,
        seed=None,
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
        self.indices = np.arange(len(self))

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
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.key]
        # get the number of images to compose with, sampled from an exponential
        # decay distribution
        n_composites = int(np.random.exponential(self.scale) + 1)
        # choose the images randomly
        chosen = np.random.choice(self.indices, size=n_composites)
        if n_composites != 1:
            chosen.sort()
        X = self.dataset[chosen]
        # if we have multiple images, flatten to a single composite
        # so that the dimensions are H x W expected by PyAbel
        if X.ndim == 3:
            X = X.sum(axis=0)
        # if we have a compose pipeline defined, run it
        if self.transform:
            return (self.transform(X), torch.FloatTensor(X).unsqueeze(0))
        # otherwise, just make a channel dimension
        X = torch.FloatTensor(X).unsqueeze(0)
        return (X, X)


class SelectiveComposite(H5Dataset):
    def __init__(self, path: str, key: str, transform=None):
        super().__init__(path, key, transform)

    def __getitem__(self, indices):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.key]
        indices = sorted(indices)
        X = self.dataset[indices].sum(axis=0).astype(float)
        if self.transform:
            return (self.transform(X), torch.FloatTensor(X).unsqueeze(0))
        X = torch.FloatTensor(X).unsqueeze(0)
        return (X, X)
