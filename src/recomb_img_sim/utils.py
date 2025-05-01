import numpy as np


def repeat_array_to_dimensions(a, dimensions):
    """Repeats a flat or ndarray along 1 or more axes.

    Args:
        a: Array to be expanded.
        dimensions: Tuple of ints defining the shape of the final array.
    Returns:
        a: Array with new dimensions.
    """
    for i in range(len(dimensions)):
        a = a[..., None].repeat(dimensions[i], len(a.shape))
    return a


def get_alt_parent(parent: str) -> str:
    """Given one parent, return the other."""
    match parent:
        case "p1":
            return "p2"
        case "p2":
            return "p1"
        case _:
            raise ValueError(f"Invalid parent {parent}")


def make_location_array(
    n_imgs: int, bin_length: int, read_depth: int
) -> np.ndarray:
    """Make an array of the indices of bin_length.

    The final array will be range(bin_length) repeated over the other
    array dimensions in the shape (n_imgs, bin_length, read_depth,
    n_channels). This will be used to determine which locations contain
    a breakpoint or are to the left or right of a breakpoint.

    Args:
        n_imgs (int): Number of images being simulated.
        bin_length (int): Length of bins in # of SNPs.
        read_depth (int): Maximum number of reads per image.
        n_channels (int): Number of color channels.

    returns:
        locations (np.ndarray, dtype=int): Array of range(bin_length) repeated
            to match the dimensions of the image array.
    """
    locations = np.arange(bin_length)
    locations = repeat_array_to_dimensions(locations, (n_imgs, read_depth))
    locations = np.moveaxis(locations, 0, 1)
    return locations


def mask_event_reads(
    n_imgs: int, bin_length: int, read_depth: int
) -> np.ndarray:
    """Filter for reads that contain events (crossovers or noncrossovers).

    In each case the event reads should be visible 50% of the time (ignoring
    the backcross reads, which are added in later and will write over some
    of this data).

    Args:
        n_imgs (int): Number of images being simulated.
        bin_length (int): Length of bins (# SNPs/image).
        read_depth (int): Max depth for each image.
        n_channels (int): Number of color channels.

    Returns:
        event_reads (np.ndarray, dtype=bool): Array where reads containing
            an event (crossover or noncrossover) = True
            (arr[img, :, event_read] = True).
    """
    rng = np.random.default_rng()
    p_event_read = rng.uniform(size=(n_imgs, read_depth))
    p_event_read = repeat_array_to_dimensions(p_event_read, (bin_length,))
    p_event_read = np.moveaxis(p_event_read, 1, 2)
    event_reads = p_event_read >= 0.5
    return event_reads


def get_breakpoint_locations(
    n_imgs: int, bin_length: int, read_depth: int
) -> np.ndarray:
    """Determine where breakpoints are for each image.

    Args:
        n_imgs (int): Number of images being simulated.
        bin_length (int): Length of bins (# SNPs/image).
        read_depth (int): Max depth for each image.
        n_channels (int): Number of color channels.

    Returns:
        breakpoint_locations (np.ndarray, dtype=int): 3D array of same
            shape as the image array with the locations of the breakpoints
            repeated across the whole image (e.g., if the breakpoint is at
            SNP 23, arr[img, ...] = 23).
    """
    rng = np.random.default_rng()
    breakpoint_locations = rng.integers(0, high=bin_length, size=n_imgs)
    breakpoint_locations = repeat_array_to_dimensions(
        breakpoint_locations, (bin_length, read_depth)
    )
    return breakpoint_locations


def make_img_array(
    n_imgs_per_class: int,
    bin_length: int,
    read_depth: int,
    n_color_channels: int,
) -> np.ndarray:
    """Initialize an array of images for a specific data class."""
    return np.zeros(
        (n_imgs_per_class, bin_length, read_depth, n_color_channels)
    )
