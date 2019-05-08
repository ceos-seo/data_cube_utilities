import numpy as np
import xarray as xr
from .clean_mask import create_circular_mask
from skimage.filters.rank import modal
from skimage.morphology import remove_small_objects

def lone_object_filter(image, min_size=2, connectivity=1, kernel_size=3):
    """
    Replaces isolated, contiguous regions of values with values representing
    the surrounding pixels.
    More specifically, this reduces noise in a raster by setting
    contiguous regions of values of a specified maximum size to
    the modal value in a specified neighborhood.
    The default argument values filter out lone, singular pixels.
    This filter is not idempotent, so it may need to be applied repeatedly
    until the output stops changing or the results are acceptable.

    Args:
        image (numpy.ndarray):
            The image to filter.
        min_size (int):
            Defines the minimum number of contiguous pixels that
            will not be filtered out.
        connectivity (int):
            The maximum distance between any two pixels such that they are
            considered one group. For example, a connectivity of 1 considers
            only adjacent values to be within one group, but a connectivity of 2
            also considers diagonally connected values to be within one group.
        kernel_size (int or float):
            The diameter of the circular kernel to use for the modal filter.
            If there are still large, contiguous regions of lone pixels,
            increase this value to remove them. Note that the larger this value
            is, the more detail may be lost in the image.

    Returns:
        The filtered image.

    Authors:
        Andrew Lubawy (andrew.m.lubawy@ama-inc.com)\n
        John Rattz    (john.c.rattz@ama-inc.com)
    """
    # median_filtered = median_filter(image, size=median_size)
    modal_filtered = modal(image, create_circular_mask(kernel_size, kernel_size))

    da = xr.DataArray(image)
    for i, val in enumerate(np.unique(image)):
        layer = remove_small_objects(image == val, min_size=min_size, connectivity=connectivity)
        filtered = da.where(layer) if i == 0 else filtered.combine_first(da.where(layer))
    filtered.values[np.isnan(filtered.values)] = modal_filtered[np.isnan(filtered.values)]
    return filtered.values