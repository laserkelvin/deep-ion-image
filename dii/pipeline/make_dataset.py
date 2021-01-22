"""
make_dataset.py

This module will implement methods to synthesize the
data used for the project.
"""

from typing import Tuple

import h5py
import numpy as np
import abel
from loguru import logger
from scipy.special import eval_legendre
from scipy import stats
from numba import njit
from tqdm.auto import tqdm
from joblib import Parallel, delayed


def pol2cart(
    angle: np.ndarray, velocity: np.ndarray, image_center=400
) -> Tuple[np.ndarray, np.ndarray]:
    x = velocity * np.cos(angle)
    y = velocity * np.sin(angle)
    return (x + image_center).astype(int), (y + image_center).astype(int)


def generate_angular_distribution(beta: float) -> Tuple[np.ndarray, np.ndarray]:
    # evaluate legendre polynomials on a grid of angles
    angles = np.linspace(0.0, 2 * np.pi, 3000)
    leg_poly = eval_legendre(2, np.cos(angles))
    return angles, (1 + beta * leg_poly)


def generate_ion_image(dim=1000, nions=10000, sigma=5.0, beta=0.0) -> np.ndarray:
    center = dim // 2
    # This sets the kinetic energy to be randomly between some
    # tolerance value
    ker_norm = np.random.uniform(low=0.0, high=center - 10.0)
    ker_distribution = np.random.normal(ker_norm, sigma, size=nions)
    theta, p_theta = generate_angular_distribution(beta)
    # Make likelihoods sum to one
    p_theta = p_theta / p_theta.sum()
    # Sample theta from p_theta
    theta_distribution = np.random.choice(theta, size=nions, p=p_theta)
    # Now ker_distribution and theta_distribution make up the complete set of data
    # we just need to convert into x,y coordinates
    x, y = pol2cart(theta_distribution, ker_distribution, image_center=center)
    image, _, _ = np.histogram2d(x, y, dim, density=False)
    return image


def generate_image(
    r_mu: float, r_sigma: float, beta: float, img_size: int = 500
) -> np.ndarray:
    # define image center
    center = img_size // 2
    # this sets the grid of radial values, by shifting to image center
    r_grid = np.arange(img_size) - center
    # generate a memory efficient xy pair representation
    x, y = np.meshgrid(r_grid, r_grid, sparse=True)
    # calculate radius and angle. The ordering for arctan2 is very important;
    # if x and y are flipped then the polarization/beta does too
    r = np.hypot(x, y)
    theta = np.arctan2(x, y)
    # arctan2 will flip the angle sign depending on the quadrant
    # this makes it so that the angle is out of 2pi
    theta[theta <= 0.0] += 2 * np.pi
    # calculate the radial distribution using a Gaussian; flatten will vectorize this
    # and reshaping it after will get you back your 2D
    p_r = stats.norm.pdf(r, loc=r_mu, scale=r_sigma)
    # calculate the angular distribution, and normalize
    p_theta = 4 * np.pi * -1 * (1 + beta * eval_legendre(2, np.cos(theta)))
    p_theta /= np.sum(p_theta)
    # image given as the product of the two probability distributions
    img = p_r * p_theta
    # normalize pixel intensities; add a small number to prevent
    # explosion
    img /= (img.max() + 1e-9)
    # now do the forward Abel transform to get the "experimental image"
    forward_abel = abel.Transform(
        img, direction="forward", method="hansenlaw"
    ).transform
    return (img, forward_abel)


def create_ion_image_composite(
    filepath: str, n_images=300000, dim=500, max_sigma=50.0, seed=42, n_jobs=1
) -> np.ndarray:
    logger.add("LOG")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Generating {n_images} ion images.")
    rng = np.random.default_rng(seed)
    # generate the range of beta parameters -1 to +2.
    betas = rng.uniform(-1.0, 2.0, size=n_images)
    sigma = rng.uniform(0.1, max_sigma, size=n_images)
    center = dim // 2
    mu = rng.uniform(0.0, center * 0.9, size=n_images)
    h5_file = h5py.File(filepath, mode="a")
    beta_values = h5_file.create_dataset(
        "beta", (n_images,), dtype=np.float32, data=betas
    )
    mu_values = h5_file.create_dataset(
        "centers", (n_images,), dtype=np.float32, data=mu
    )
    # true_images = h5_file.require_dataset("true", (n_images, dim, dim), dtype=np.float32, compression="gzip", compression_opts=1)
    # projections = h5_file.require_dataset("projection", (n_images, dim, dim), dtype=np.float32, compression="gzip", compression_opts=1)
    try:
        # serial processing
        if n_jobs == 1:
            for index, (mu_i, sigma_i, beta_i) in tqdm(
                enumerate(zip(mu, sigma, betas))
            ):
                true_image, projection = generate_image(mu_i, sigma_i, beta_i, dim)
                true_images[index, :, :] = true_image
                projections[index, :, :] = projection
        elif n_jobs > 1:
            data = Parallel(n_jobs=n_jobs)(
                delayed(generate_image)(mu_i, sigma_i, beta_i, dim)
                for mu_i, sigma_i, beta_i in tqdm(zip(mu, sigma, betas))
            )
            temp_true, temp_projection = list(), list()
            for image_pair in data:
                true_image, projection = image_pair
                temp_true.append(true_image)
                temp_projection.append(projection)
            true_images = h5_file.create_dataset(
                "true",
                (n_images, dim, dim),
                dtype=np.float32,
                data=np.array(temp_true, dtype=np.float32),
            )
            projections = h5_file.create_dataset(
                "projection",
                (n_images, dim, dim),
                dtype=np.float32,
                data=np.array(temp_projection, dtype=np.float32),
            )
        # do the train/test/dev split; 10% dev, 10% test, and 80% train
        indices = np.arange(n_images, dtype=int)
        rng.shuffle(indices)
        short_num = int(n_images * 0.1)
        dev = indices[:short_num]
        test = indices[short_num : short_num * 2]
        train = indices[short_num * 2 :]
        h5_file.create_dataset("train", (len(train),), dtype=np.uint16, data=train)
        h5_file.create_dataset("dev", (len(dev),), dtype=np.uint16, data=dev)
        h5_file.create_dataset("test", (len(test),), dtype=np.uint16, data=test)
    finally:
        h5_file.close()
