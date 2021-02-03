
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import h5py

seed = 42
rng = np.random.default_rng(seed)

h5_data = h5py.File("128_ion_images.h5", "r")
n_images = len(h5_data["true"])
indices = np.arange(n_images)

N_SELECT = 100
n_row = int(np.sqrt(N_SELECT))
chosen = rng.choice(indices, replace=False, size=N_SELECT)
chosen.sort()

fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row))

images = h5_data["true"][chosen]

for ax, im in zip(grid, images):
    ax.imshow(im)

fig.tight_layout()
fig.savefig("true-images.png", dpi=300)
