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


def stats_filter(dataarray, statistic, filter_shape=(1, 1)):
    """
    Returns a mean, median, or standard deviation filter of an `xarray.DataArray`.
    This function is more accurate than using SciPy or scikit-image methods, because
    those don't handle the extremities ideally (edges and corners).
    Specifically, only values actually inside the filter should be considered,
    so the data is padded with NaNs when "convolving" `dataarray` with the
    equally-weighted kernel of shape `filter_shape`.
    This function is resilient to NaNs.

    Parameters
    ----------
    dataarray: xarray.DataArray
        The data to create a filtered version of. Must have 3 dimensions, with
        the last being 'time'.
    statistic: string
        The name of the statistic to use for the filter.
        The possible values are ['mean', 'median', 'std'].
    filter_shape: list-like of 2 odd, positive integers
        The shape of the filter to use. Both dimensions should have odd lengths.
    """
    filter_dims = dataarray.dims[:2]
    filter_coords = {dim: dataarray.coords[dim] for dim in filter_dims}
    filter_output = xr.DataArray(np.full(dataarray.shape[:2], np.nan),
                                 coords=filter_coords, dims=filter_dims)
    if filter_shape == (1, 1):
        agg_func_kwargs = dict(a=dataarray.values, axis=dataarray.get_axis_num('time'))
        if statistic == 'mean':
            filter_output.values[:] = np.nanmean(**agg_func_kwargs)
        elif statistic == 'median':
            filter_output.values[:] = np.nanmedian(**agg_func_kwargs)
        elif statistic == 'std':
            filter_output.values[:] = np.nanstd(**agg_func_kwargs)
    else:
        # Allocate a Numpy array containing the content of `dataarray`, but padded
        # with NaNs to ensure the statistics are correct at the x and y extremeties of the data.
        flt_shp = np.array(filter_shape)
        del filter_shape
        shp = np.array(dataarray.shape[:2])
        pad_shp = (*(shp + flt_shp - 1), dataarray.shape[2])
        padding = (flt_shp - 1) // 2  # The number of NaNs from an edge of the padding to the data.
        padded_arr = np.full(pad_shp, np.nan)
        padded_arr[padding[0]:pad_shp[0] - padding[0],
        padding[1]:pad_shp[1] - padding[1]] = dataarray.values

        # For each point in the first two dimensions of `dataarray`...
        for i in range(filter_output.shape[0]):
            for j in range(filter_output.shape[1]):
                padded_arr_segment = padded_arr[i:i + flt_shp[0],
                                     j:j + flt_shp[1]]
                if statistic == 'mean':
                    filter_output.values[i, j] = np.nanmean(padded_arr_segment)
                elif statistic == 'median':
                    filter_output.values[i, j] = np.nanmedian(padded_arr_segment)
                elif statistic == 'std':
                    filter_output.values[i, j] = np.nanstd(padded_arr_segment)

    return filter_output
