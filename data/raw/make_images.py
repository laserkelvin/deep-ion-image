
from dii.pipeline import make_dataset


N_IMAGES = 50
FILEPATH = "ion_images.h5"

# Square image dimensions
DIM = 1200
MAX_IONS = 500000

SEED = 42

make_dataset.create_ion_image_composite(FILEPATH, N_IMAGES, DIM, MAX_IONS, seed=SEED)

