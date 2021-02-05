
"""
get_benchmark_set.py

This script is used to extract a set of data from the pipeline
in order to be used as a point of comparison between CNN models
and conventional reconstruction algorithms.
"""

import h5py
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from dii.pipeline.datautils import CompositeH5Dataset

PATH = "128_benchmark_set.h5"
N_IMAGES = 128
SEED = 69420

with h5py.File("../raw/128_ion_images.h5", "r") as read_file:
    indices = read_file["dev"][:]

dataset = CompositeH5Dataset("../raw/128_ion_images.h5", "projection", seed=69420, indices=indices)
loader = DataLoader(dataset, batch_size=N_IMAGES, num_workers=1)

batch = next(iter(loader))
projections, centrals, _, _ = batch

# convert images to PIL format
# pil_projections = ToPILImage()(projections)
# pil_centrals = ToPILImage()(centrals)

transform = ToPILImage()

# PIL save to JPG pipeline. This makes things easier to look at as well as 
# for the methods that don't take other formats
for folder, image_type in zip(["inputs", "outputs"], [projections, centrals]):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for index, image in enumerate(image_type):
        image = transform(image)
        image.save(f"{folder}/{index}.jpg")


with h5py.File(PATH, "a") as write_file:
    # convert torch tensors to numpy arrays
    projections = projections.cpu().numpy()
    centrals = centrals.cpu().numpy()
    write_file.create_dataset("inputs", data=projections)
    write_file.create_dataset("outputs", data=centrals)
    for folder, image_type in zip(["inputs", "outputs"], [projections, centrals]):
        for index, image in enumerate(image_type):
            # squeeze the "channel" dimension
            np.savetxt(f"{folder}/{index}.dat", image[0])