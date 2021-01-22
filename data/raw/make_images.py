
from dii.pipeline import make_dataset


N_IMAGES = 10000
FILEPATH = "ion_images.h5"

# Square image dimensions
DIM = 500

SEED = 42

make_dataset.create_ion_image_composite(FILEPATH, N_IMAGES, dim=DIM, seed=SEED, n_jobs=8)

