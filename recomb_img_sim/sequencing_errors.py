import numpy as np

from utils import *


def filter_errors(
    n_imgs: np.ndarray, bin_length: int, read_depth: int, error_rate: float
) -> np.ndarray:
    """Get locations of sequencing errors."""
    rng = np.random.default_rng()
    p_error = rng.uniform(size=(n_imgs, bin_length, read_depth))
    error_mask = p_error < error_rate
    return error_mask


def mask_errors_matching_parent(
    n_imgs: np.ndarray,
    bin_length: int,
    read_depth: int,
    error_mask: np.ndarray,
) -> tuple:
    """Determine which sequencing errors match one of the parents."""
    rng = np.random.default_rng()
    p_parent_match = rng.uniform(size=(n_imgs, bin_length, read_depth))
    p1_errors = error_mask & (p_parent_match < 0.25)
    p2_errors = error_mask & (p_parent_match >= 0.25) & (p_parent_match < 0.5)
    unique_errors = error_mask & (p_parent_match >= 0.5)
    return p1_errors, p2_errors, unique_errors


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
