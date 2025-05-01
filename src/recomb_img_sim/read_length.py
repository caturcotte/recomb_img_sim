import numpy as np

from .utils import *


def get_read_centers(n_imgs, read_depth, bin_length):
    """Get the centers of each read in each image."""
    rng = np.random.default_rng()
    read_centers = rng.integers(0, high=bin_length, size=(n_imgs, read_depth))
    read_centers = repeat_array_to_dimensions(read_centers, (bin_length,))
    read_centers = np.moveaxis(read_centers, 2, 1)
    return read_centers


def get_read_lengths(
    read_length_mean, read_length_stdev, n_imgs, read_depth, bin_length
):
    """Get the lengths of each read in each image."""
    rng = np.random.default_rng()
    read_lengths = rng.normal(
        loc=read_length_mean,
        scale=read_length_stdev,
        size=(n_imgs, read_depth),
    )
    read_lengths = repeat_array_to_dimensions(read_lengths, (bin_length,))
    read_lengths = np.moveaxis(read_lengths, 2, 1)
    return read_lengths


def get_read_starts_and_ends(read_centers, read_lengths):
    """Find the starts and ends of reads based on the centers and lengths."""
    read_starts = read_centers - (read_lengths / 2)
    read_starts = np.sort(read_starts)
    read_ends = read_starts + read_lengths
    return read_starts, read_ends


def find_out_of_bounds_locations(locations, read_starts, read_ends):
    """Mask locations that are beyond the read starts and ends."""
    pre_read_starts = locations < read_starts
    post_read_ends = locations > read_ends
    beyond_read_length = pre_read_starts | post_read_ends
    return beyond_read_length


def shorten_reads(
    imgs, missing_data_color, read_length_mean, read_length_stdev
):
    """Remove data from locations that are beyond read starts and ends."""
    n_imgs, bin_length, read_depth = imgs.shape[:-1]
    locations = make_location_array(n_imgs, bin_length, read_depth)
    read_centers = get_read_centers(n_imgs, bin_length, read_depth)
    read_lengths = get_read_lengths(
        read_length_mean, read_length_stdev, n_imgs, bin_length, read_depth
    )
    read_starts, read_ends = get_read_starts_and_ends(
        read_centers, read_lengths
    )
    out_of_bounds = find_out_of_bounds_locations(
        locations, read_starts, read_ends
    )
    imgs[out_of_bounds, :] = missing_data_color
    return imgs
