import numpy as np

from utils import *


def mask_sequencing_errors(n_imgs, bin_length, read_depth, error_rate):
    shape = (n_imgs, bin_length, read_depth)
    rng = np.random.default_rng()
    p_error = rng.uniform(size=shape)
    error_mask = p_error < error_rate
    error_mask = repeat_array_to_dimensions(error_mask, shape)
    return error_mask


def mask_errors_matching_parent(n_imgs, bin_length, read_depth, error_mask):
    rng = np.random.default_rng()
    p_parent_match = rng.uniform(size=(n_imgs, bin_length, read_depth))
    p1_errors = error_mask & (p_parent_match < 0.25)
    p2_errors = error_mask & (p_parent_match >= 0.25) & (p_parent_match < 0.5)
    unique_errors = error_mask & (p_parent_match >= 0.5)
    return p1_errors, p2_errors, unique_errors


def filter_errors(
    n_imgs,
    bin_length,
    read_depth,
    error_rate
):
    rng = np.random.default_rng()
    p_error = rng.uniform(size=(n_imgs, bin_length, read_depth))
    error_mask = p_error < error_rate
    return error_mask

def make_sequencing_errors(
    imgs: np.ndarray, colors: dict, error_rate: float
) -> np.ndarray:
    """
    Generates random sequencing errors in images.

    Args:
        imgs:
            Array of shape (n_images, bin_length, read_depth, n_color_channels).
        colors:
            Dict of colors to be used for
        error_rate:
            Rate of sequencing errors.
    """
    n_imgs, bin_length, read_depth = imgs.shape[:-1]
    errors = filter_errors(n_imgs, bin_length, read_depth, error_rate)
    p1_errors, p2_errors, unique_errors = mask_errors_matching_parent(
        n_imgs, bin_length, read_depth, errors
    )
    imgs[p1_errors, :] = colors["p1"]
    imgs[p2_errors, :] = colors["p2"]
    imgs[unique_errors, :] = colors["missing_data"]
    return imgs
