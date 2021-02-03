
from dii.pipeline import make_dataset

N_IMAGES = 30000
FILEPATH = "128_ion_images.h5"

# Square image dimensions
DIM = 128

SEED = 1296516

make_dataset.create_ion_image_composite(FILEPATH, N_IMAGES, dim=DIM, seed=SEED, n_jobs=8)

