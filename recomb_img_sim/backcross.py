import numpy as np

from utils import *


def make_backcross_reads(
    imgs: np.ndarray, backcross_parent_color: tuple[int]
) -> np.ndarray:
    """Add in reads from the backcross parent."""
    n_imgs, bin_length, read_depth = imgs.shape[:-1]
    rng = np.random.default_rng()
    p_backcross_reads = rng.uniform(size=(n_imgs, read_depth))
    p_backcross_reads = np.repeat(
        p_backcross_reads[..., None], bin_length, axis=-1
    )
    p_backcross_reads = np.moveaxis(p_backcross_reads, 2, 1)
    backcross_reads = p_backcross_reads >= 0.5
    imgs[backcross_reads, :] = backcross_parent_color
    return imgs
