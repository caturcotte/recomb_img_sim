import numpy as np

from .utils import *


def make_read_index_array(n_imgs, bin_length, read_depth):
    """Make an array for the indices of the read depth axis.

    The resulting array should be shape (n_imgs, bin_length, read_depth).
    """
    reads = np.arange(read_depth)
    reads = repeat_array_to_dimensions(reads, (n_imgs, bin_length))
    reads = np.moveaxis(reads, 0, -1)
    return reads


def get_missing_reads(
    n_reads,
    n_imgs,
    bin_length,
    read_depth,
):
    """Determine how many reads this image has.

    Missing reads are reads exceeding the read depth of each image.
    """
    reads = make_read_index_array(n_imgs, bin_length, read_depth)
    empty_reads = reads > n_reads
    return empty_reads


def remove_some_reads(
    imgs: np.ndarray,
    missing_data_color: tuple,
    mean_read_depth,
    read_depth_std,
):
    """Turn reads exceeding the read depth of each image into missing data."""
    n_imgs, bin_length, read_depth = imgs.shape[:3]
    rng = np.random.default_rng()
    n_reads = rng.normal(
        loc=mean_read_depth, scale=read_depth_std, size=n_imgs
    )
    n_reads = repeat_array_to_dimensions(n_reads, (bin_length, read_depth))
    empty_reads = get_missing_reads(n_reads, *imgs.shape[:-1])
    imgs[empty_reads, :] = missing_data_color
    return imgs
