
from dii.pipeline import make_dataset

FILEPATH = "128_grid_images.h5"

# Square image dimensions
DIM = 128

make_dataset.create_ion_image_grid(FILEPATH, dim=DIM, n_jobs=8)

