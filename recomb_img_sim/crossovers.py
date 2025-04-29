import numpy as np

from utils import *


def mask_breakpoint_to_end_of_read(
    co_reads: np.ndarray,
    co_locations: np.ndarray,
) -> tuple:
    """Filter for locations from the crossover to the end of the bin length.

    Args:
        co_reads: A 4D bool array with the same shape as the image array
            (n_imgs, bin_length, read_depth, n_channels) that indicates which
            reads contain crossovers (arr[img, :, co_read, :] = True).
        co_locations: A 4D int array of the same shape as the image array that
            lists the crossover location for each image
            (arr[img, co_site, :, :] = True).

    Returns:
        left_of_brk_on_co_reads: 4D bool array that is True for locations left
            of the breakpoint on reads with a crossover.
        right_of_brk_on_co_reads: 4D bool array that is True for locations
            right of the breakpoint on reads with a crossover.
        right_of_brk_on_other_reads: 4D bool array that is True for locations
            right of the breakpoint on reads that do not have a crossover
            (required to model reciprocal exchanges).
    """
    n_imgs, bin_length, read_depth = co_reads.shape
    locations = make_location_array(n_imgs, bin_length, read_depth)
    right_of_brk = locations >= co_locations
    left_of_brk = ~right_of_brk
    left_of_brk_on_co_reads = co_reads & left_of_brk
    right_of_brk_on_co_reads = co_reads & right_of_brk
    left_of_brk_on_other_reads = ~co_reads & left_of_brk
    right_of_brk_on_other_reads = ~co_reads & right_of_brk
    return (
        left_of_brk_on_co_reads,
        right_of_brk_on_co_reads,
        left_of_brk_on_other_reads,
        right_of_brk_on_other_reads,
    )


def mask_locations_distal_to_crossovers(
    left_breakpoint_mask: np.ndarray,
    right_breakpoint_mask: np.ndarray,
) -> np.ndarray:
    """ Filter for locations distal to the crossover on the chromosome arm.

    Here distal has a 50% probability of being left of the crossover or right
    of the crossover, we just want to simulate both scenarios so that we can
    train the model on both.

    Biologically the "sidedness" of the image would depend on the crossover
    either being on the left or right chromosome arm and whether the NDJ event
    was meiosis I or meiosis II. For MI, the chromosome will be homozygous
    distal to the crossover, and for MII it will be heterozygous distal to the
    crossover. So a MI nondisjunction on chr2L will be homozygous on the left
    and heterozygous on the right, but a MII nondisjunction on chr2L would be
    heterozygous on the left and homozygous on the right.

    Args:
        left_breakpoint_mask: 4D bool array that is True for locations left of
            the crossover on crossover-containing reads.
        right_breakpoint_mask: 4D bool array that is True for locations right
            of the crossover on crossover-containing reads.

    Returns:
        distal_mask: 4D bool array that is True for locations distal to the
            crossover on crossover-containing reads.
    """
    n_imgs, bin_length, read_depth = left_breakpoint_mask.shape
    rng = np.random.default_rng()
    p_co_direction = rng.uniform(size=n_imgs)
    p_co_direction = repeat_array_to_dimensions(
        p_co_direction, (bin_length, read_depth)
    )
    left_co_mask = p_co_direction > 0.5
    distal_mask_l = left_breakpoint_mask & left_co_mask
    distal_mask_r = right_breakpoint_mask & ~left_co_mask
    distal_mask = distal_mask_l | distal_mask_r
    return distal_mask


def make_cos(imgs: np.ndarray, colors: dict, co_type: str) -> np.ndarray:
    """Make crossovers in the image array.

    Args:
        imgs: 4D int array of shape (n_imgs, bin_length, read_depth,
            n_channels) containing colors for each pixel in every image for
            this data class.
        colors (dict[str: tuple]): The RGB values to use for each parent.
        co_type (str): Type of crossover to model (p1, p2 or reciprocal).

    Returns:
        imgs: 4D array with crossovers modeled.

    Raises:
        ValueError: if co_type is not p1, p2 or reciprocal
            (this should already be checked by the JSON schema anyway).
    """
    print("Getting crossover locations...")
    co_locations = get_breakpoint_locations(*imgs.shape[:-1])
    print("Determining which reads contain crossovers...")
    co_reads = mask_event_reads(*imgs.shape[:-1])
    (
        left_of_brk_on_co_reads,
        right_of_brk_on_co_reads,
        left_of_brk_on_other_reads,
        right_of_brk_on_other_reads,
    ) = mask_breakpoint_to_end_of_read(co_reads, co_locations)
    distal_on_co_reads = mask_locations_distal_to_crossovers(
        left_of_brk_on_co_reads, right_of_brk_on_co_reads
    )
    print("Applying crossovers to images...")
    match co_type:
        case "reciprocal":
            imgs[right_of_brk_on_co_reads, :] = colors["p1"]
            imgs[left_of_brk_on_co_reads, :] = colors["p2"]
            imgs[left_of_brk_on_other_reads, :] = colors["p1"]
            imgs[right_of_brk_on_other_reads, :] = colors["p2"]
        case "p1" | "p2":
            alt_parent = get_alt_parent(co_type)
            imgs[~distal_on_co_reads, :] = colors[co_type]
            imgs[distal_on_co_reads, :] = colors[alt_parent]
        case _:
            raise ValueError(f"Invalid crossover type: {co_type}")
    return imgs
