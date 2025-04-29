import numpy as np

from utils import *


def mask_nco_site(
    n_imgs: int,
    bin_length: int,
    read_depth: int,
    nco_read_mask: np.ndarray,
    nco_locations: np.ndarray,
) -> np.ndarray:
    """Filter for the site of noncrossovers in the image array.

    Args:
        n_imgs (int): Number of images being simulated.
        bin_length (int): Size of each bin in number of SNPs.
        read_depth (int): Maximum read depth for each image.
        nco_read_mask (np.ndarray[bool]): 3D bool array where reads 
            containing NCOs are True (arr[img, :, read] = True).
        nco_locations: 3D boolean array with the NCO location for each image
            (arr[img, co_location, :] = True).

    Returns:
        nco_mask: 4D bool array where sites of NCOs on reads with NCOs are
        True (arr[img, co_location, read, :] = True).
    """
    locations = make_location_array(n_imgs, bin_length, read_depth)
    nco_location_mask = locations == nco_locations
    nco_mask = nco_location_mask & nco_read_mask
    return nco_mask


def put_nco_colors_in_array(
    imgs: np.ndarray,
    nco_mask: np.ndarray,
    nco_read_mask: np.ndarray,
    colors: dict,
    parent: str,
    alt_parent: str,
    background: str
) -> np.ndarray:
    """Put the noncrossovers in the image array.
    
    Args:
        imgs (np.ndarray): The image array.
        nco_mask (np.ndarray): 3D bool array with sites of NCOs on
            NCO reads = True (arr[img, nco_site, nco_read, :] = True).
        nco_read_mask (np.ndarray): 4D array of the same shape as imgs with
            NCO reads = True (arr[img, :, nco_read, :] = True).
        colors (dict[str: tuple[int]]): Dict containing the colors to use for
            each parent.
        parent (str): The parental genotype for the noncrossover.
        alt_parent (str): The other parent.
        background (str): The background genotype (heterozygous or homozygous).

    Raises:
        ValueError: If background isn't heterozygous or homozygous.

    Returns:
        imgs (np.ndarray): The image array, now with noncrossovers added.
    """
    match background:
        case "homozygous":
            imgs[:, :, :, :] = colors[alt_parent]
            imgs[nco_mask, :] = colors[parent]
        case "heterozygous":
            imgs[nco_read_mask, :] = colors[alt_parent]
            imgs[nco_mask, :] = colors[parent]
            imgs[~nco_read_mask, :] = colors[parent]
        case _:
            raise ValueError(f"Invalid background {background}")
    return imgs


def make_ncos(
    imgs: np.ndarray,
    colors: dict,
    parent: str,
    background: str,
) -> np.ndarray:
    """Make noncrossovers in the image array.

    Args:
        imgs (np.ndarray, dtype=int): A 4D array containing all of the images
            to be simulated for this data class.
        colors (dict[str: tuple[int]]): The colors to be used for each of the
            parents.
        parent (str): The parental genotype of the noncrossover.
        background (str): The background genotype (heterozygous or
            homozygous).

    Returns:
        imgs: The image array now with noncrossovers added.
    """
    n_imgs, bin_length, read_depth = imgs.shape[:-1]
    nco_locations = get_breakpoint_locations(n_imgs, bin_length, read_depth)
    nco_read_mask = mask_event_reads(n_imgs, bin_length, read_depth)
    nco_mask = mask_nco_site(
        n_imgs,
        bin_length,
        read_depth,
        nco_read_mask,
        nco_locations,
    )
    alt_parent = get_alt_parent(parent)
    put_nco_colors_in_array(
        imgs, nco_mask, nco_read_mask, colors, parent, alt_parent, background
    )
    return imgs
