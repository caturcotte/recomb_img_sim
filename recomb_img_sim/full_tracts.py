import numpy as np


def make_heterozygous(imgs: np.ndarray, colors: dict) -> np.ndarray:
    """Make image class for heterozygous genomic regions."""
    n_imgs, bin_length, read_depth = imgs.shape[:-1]
    rng = np.random.default_rng()
    p_parent2 = rng.uniform(size=(n_imgs, read_depth))
    p_parent2 = p_parent2[..., None].repeat(bin_length, axis=2)
    p_parent2 = np.moveaxis(p_parent2, [0, 1, 2], [0, 2, 1])
    p2_reads = p_parent2 >= 0.5
    imgs[p2_reads, :] = colors["p2"]
    imgs[~p2_reads, :] = colors["p1"]
    return imgs


def make_homozygous(imgs, parent_color):
    """Make image class for homozygous genomic regions."""
    imgs[..., :] = parent_color
    return imgs
