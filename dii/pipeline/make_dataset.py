
"""
make_dataset.py

This module will implement methods to synthesize the
data used for the project.
"""

from typing import Tuple

import h5py
import numpy as np
import abel
from scipy.special import eval_legendre
from numba import njit


def pol2cart(angle: np.ndarray, velocity: np.ndarray, image_center=400) -> Tuple[np.ndarray, np.ndarray]:
    x = velocity * np.cos(angle)
    y = velocity * np.sin(angle)
    return (x + image_center).astype(int), (y + image_center).astype(int)


def generate_angular_distribution(beta: float) -> Tuple[np.ndarray, np.ndarray]:
    # evaluate legendre polynomials on a grid of angles
    angles = np.linspace(0., 2 * np.pi, 3000)
    leg_poly = eval_legendre(2, np.cos(angles))
    return angles, (1 + beta * leg_poly)


def generate_ion_image(dim=1000, nions=10000, sigma=5., beta=0.) -> np.ndarray:
    center = dim // 2
    # This sets the kinetic energy to be randomly between some
    # tolerance value
    ker_norm = np.random.uniform(low=10., high=center - 10.)
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


def create_ion_image_composite(filepath: str, n_images=300000, dim=1200, max_ions=500000, max_sigma=10., seed=42) -> np.ndarray:
    seed = np.random.seed(seed)
    n_ions = np.random.randint(10000, max_ions, size=n_images)
    # generate the range of beta parameters -1 to +2.
    betas = (np.random.rand(n_images) * 3.) - 1.
    sigmas = np.random.rand(n_images) * 10.
    h5_file = h5py.File(filepath, mode="a")
    images = h5_file.require_dataset("true", (n_images, dim, dim), dtype=np.float32, compression="gzip", compression_opts=9)
    for index, (ion_count, beta, sigma) in enumerate(zip(n_ions, betas, sigmas)):
        true_image = generate_ion_image(dim, ion_count, sigma, beta)
        images[index,:,:] = true_image


