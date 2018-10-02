import numpy as np
import xarray as xr
from .clean_mask import landsat_qa_clean_mask, landsat_clean_mask_invalid
from xarray.ufuncs import logical_and as xr_and
from .sort import xarray_sortby_coord

def xarray_concat_and_merge(*args, concat_dim='time', sort_dim='time'):
    """
    Given parameters that are each a list of `xarray.Dataset` objects, merge each list 
    into an `xarray.Dataset` object and return all such objects in the same order.

    Parameters
    ----------
    *args: list of lists of `xarray.Dataset`.
        A list of lists of `xarray.Dataset` objects to merge.
    concat_dim, sort_dim: str or list of str
        The string name of the dimension to concatenate or sort bythe data.
        If a list, must be same length as `*args`, where each element of these variables
        corresponds to the same element in `*args` by index.

    Returns
    -------
    merged: list of `xarray.Dataset`
        A tuple of the same length as `*args`, containing the merged data. 
    """
    merged = []
    for i, arg in enumerate(args):
        current_concat_dim = concat_dim[i] if isinstance(concat_dim, list) else concat_dim
        current_sort_dim = sort_dim[i] if isinstance(sort_dim, list) else sort_dim
        dataset_temp = xr.concat(arg, dim=concat_dim)
        merged.append(xarray_sortby_coord(dataset_temp, coord=sort_dim))
    return merged

def merge_datasets(datasets_temp, clean_masks_temp, masks_per_platform=None):
    """
    Merges dictionaries of platforms mapping to datasets, dataset clean masks,
    and lists of other masks into one dataset, one dataset clean mask, and one
    of each type of other mask, ordering all by time.

    Parameters
    ----------
    datasets_temp, clean_masks_temp, masks_per_platform: dict
        Maps platforms to datasets to merge, dataset masks to merge,
        and lists of masks to merge separately, respectively.
        All entries must have a 'time' dimension.

    Returns
    -------
    dataset: xarray.Dataset
        The raw data requested. Can be cleaned with `dataset.where(clean_mask)`.
    clean_mask: xarray.DataArray
        The clean mask.
    masks: list of xarray.DataArray
        A list of individual masks.

    Raises
    ------
    AssertionError: If no data was retrieved for any query (i.e. `len(datasets_temp) == 0`).
    
    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    assert len(datasets_temp) > 0, "No data was retrieved." # No data for any query.
    # If multiple non-empty datasets were retrieved, merge them and sort by time.
    masks = None
    if len(datasets_temp) > 1:
        # Merge datasets.
        datasets_temp_list = list(datasets_temp.values())
        dataset = xr.concat(datasets_temp_list, dim='time')
        dataset = xarray_sortby_coord(dataset, 'time')
        # Merge clean masks.
        clean_masks_temp_list = list(clean_masks_temp.values())
        clean_mask = xr.concat(clean_masks_temp_list, dim='time')
        clean_mask = xarray_sortby_coord(clean_mask, 'time')
        # Merge masks.
        if masks_per_platform is not None:
            num_platforms = len(masks_per_platform.keys())
            num_masks = len(list(masks_per_platform.values())[0])
            np_platform_masks = np.empty((num_platforms, num_masks), dtype=object)
            for i, mask_list in enumerate(masks_per_platform.values()):
                np_platform_masks[i] = mask_list
            masks = []
            for j in range(num_masks):
                masks.append(xr.concat(list(np_platform_masks[:,j]), dim='time'))
    else: # Select the only dataset.
        dataset = datasets_temp[list(datasets_temp.keys())[0]]
        clean_mask = clean_masks_temp[list(clean_masks_temp.keys())[0]]
        if masks_per_platform is not None:
            masks = masks_per_platform[list(masks_per_platform.keys())[0]]
    return dataset, clean_mask, masks

def load_simple(dc, platform, product, load_params={}, masking_params={}, indiv_masks=None):
    """
    Simplifies loading from the Data Cube by retrieving a dataset along with its mask.

    Parameters
    ----------
    dc: datacube.api.core.Datacube
        The Datacube instance to load data with.
    platform, product: str
        Strings denoting the platform and product to retrieve data for.
    load_params: dict, optional
        A dictionary of parameters for `dc.load()`.
        Here are some common load parameters:
        *lat, lon: list-like 2-tuples of minimum and maximum values for latitude and longitude, respectively.*
        *time: list-like     A 2-tuple of the minimum and maximum times for acquisitions.*
        *measurements: list-like The list of measurements to retrieve from the Datacube.*
        For example, to load data with different time ranges for different platforms, we could do the following:
        `{'LANDSAT_7': dict(common_load_params, **dict(time=ls7_date_range)), 
          'LANDSAT_8': dict(common_load_params, **dict(time=ls8_date_range))}`, where `common_load_params` is 
        a dictionary of load parameters common to both - most notably 'lat', 'lon', and 'measurements'.
    masking_params: dict, optional
        A dictionary of keyword arguments for corresponding masking functions.
        For example: {'cover_types':['cloud']} would retain only clouds for Landsat products, 
        because `landsat_qa_clean_mask()` is used for the Landsat family of platforms.
    indiv_masks: list
        A list of masks to return (e.g. ['water']). 
        These do not have to be the same used to create `clean_mask`.

    Returns
    -------
    dataset: xarray.Dataset
        The raw data requested. Can be cleaned with `dataset.where(clean_mask)`.
    clean_mask: xarray.DataArray
        The clean mask, formed as a logical AND of all masks used.
    masks: list of xarray.DataArray
        A list of the masks requested by `indiv_masks`, 
        or `None` if `indiv_masks` is not specified.

    Raises
    ------
    AssertionError: If no data is retrieved for any platform query.
    
    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    current_load_params = dict(platform=platform, product=product)
    current_load_params.update(load_params)
    dataset = dc.load(**current_load_params)
    assert len(dataset.dims) > 0, "No data was retrieved."
    # Get the clean mask for the appropriate LANDSAT satellite platform.
    clean_mask = landsat_qa_clean_mask(dataset, platform, **masking_params)
    # Get the mask for removing data ouside the accepted range of LANDSAT 7 and 8.
    clean_mask = xr_and(clean_mask, landsat_clean_mask_invalid(dataset))
    # Retrieve individual masks.
    if indiv_masks is None:
        masks = None
    else:
        masks = []
        for mask in indiv_masks:
            masks.append(landsat_qa_clean_mask(dataset, platform, cover_types=[mask]))
    return dataset, clean_mask, masks

def load_multiplatform(dc, platforms, products, load_params={}, masking_params={}, indiv_masks=None):
    """
    Load and merge data as well as clean masks, given a list of platforms and products.
    Currently only tested on Landsat data.
    
    Parameters
    ----------
    dc: datacube.api.core.Datacube
        The Datacube instance to load data with.
    platforms, products: list-like
        A list-like of platforms and products. Both must have the same length.
    load_params: dict, optional
        A dictionary of parameters for `dc.load()` or a dictionary of dictionaries of such parameters,
        mapping platform names to parameter dictionaries (primarily useful for selecting different time ranges).
        Here are some common load parameters:
        *lat, lon: list-like 2-tuples of minimum and maximum values for latitude and longitude, respectively.*
        *time: list-like     A 2-tuple of the minimum and maximum times for acquisitions or a list of such 2-tuples.*
        *measurements: list-like The list of measurements to retrieve from the Datacube.*
        For example, to load data with different time ranges for different platforms, we could do the following:
        `{'LANDSAT_7': dict(common_load_params, **dict(time=ls7_date_range)), 
          'LANDSAT_8': dict(common_load_params, **dict(time=ls8_date_range))}`, where `common_load_params` is 
        a dictionary of load parameters common to both - most notably 'lat', 'lon', and 'measurements'.
    masking_params: dict, optional
        A dictionary mapping platform names to a dictionary of keyword arguments for corresponding masking functions.
        For example: {'LANDSAT_7': {'cover_types':['cloud']}, 'LANDSAT_8': {'cover_types': ['cloud']}} would retain
        only clouds, because `landsat_qa_clean_mask()` is used for the Landsat family of platforms.
    indiv_masks: list
        A list of masks to return (e.g. ['water']). 
        These do not have to be the same used to create `clean_mask`.
    
    Returns
    -------
    dataset: xarray.Dataset
        The raw data requested. Can be cleaned with `dataset.where(clean_mask)`.
    clean_mask: xarray.DataArray
        The clean mask, formed as a logical AND of all masks used.
    masks: list of xarray.DataArray
        A list of the masks requested by `indiv_masks`, 
        or `None` if `indiv_masks` is not specified.
        
    Raises
    ------
    AssertionError: If no data is retrieved from any product.
    
    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    datasets_temp = {} # Maps platforms to datasets to merge.
    clean_masks_temp = {} # Maps platforms to clean masks to merge.
    masks_per_platform = {} if indiv_masks is not None else None # Maps platforms to lists of masks.
    for product,platform in zip(products, platforms):
        current_load_params = dict(platform=platform, product=product)
        current_masking_params = masking_params.get(platform, masking_params)
        
        # Handle `load_params` as a dict of dicts of platforms mapping to load params.
        if isinstance(list(load_params.values())[0], dict): 
            current_load_params.update(load_params.get(platform, {}))
        else: # Handle `load_params` as a dict of load params.
            current_load_params.update(load_params)
        # Load each time range of data.
        time = current_load_params.get('time')
        if isinstance(time[0], tuple) or \
           isinstance(time[0], list): # Handle `time` as a list of time ranges.
            datasets_time_parts = []
            clean_masks_time_parts = []
            masks_time_parts = np.empty((len(time), len(indiv_masks)), dtype=object)\
                               if indiv_masks is not None else None
            for i, time_range in enumerate(time):
                time_range_load_params = current_load_params
                time_range_load_params['time'] = time_range
                try:
                    dataset_time_part, clean_mask_time_part, masks_time_part = \
                        load_simple(dc, platform, product, time_range_load_params, 
                                    masking_params, indiv_masks)
                    datasets_time_parts.append(dataset_time_part)
                    clean_masks_time_parts.append(clean_mask_time_part)
                    if indiv_masks is not None:
                        masks_time_parts[i] = masks_time_part
                except (AssertionError):
                    continue
            datasets_temp[platform], clean_masks_temp[platform] = \
                xarray_concat_and_merge(datasets_time_parts, clean_masks_time_parts)
            if indiv_masks is not None:
                masks_per_platform[platform] = xarray_concat_and_merge(*masks_time_parts.T)
        else: # Handle `time` as a single time range.
            try:
                datasets_temp[platform], clean_masks_temp[platform], masks = \
                    load_simple(dc, platform, product, current_load_params, masking_params, indiv_masks)
                if indiv_masks is not None:
                    masks_per_platform[platform] = masks
            except (AssertionError):
                continue
    return merge_datasets(datasets_temp, clean_masks_temp, masks_per_platform)

def get_overlapping_area(api, platforms, products):
    """
    Returns the minimum and maximum latitude and longitude of the overlapping area for a set of products.
    
    Parameters
    ----------
    api: DataAccessApi
        An instance of `DataAccessApi` from `utils.data_cube_utilities`.
    platforms, products: list-like
        A list-like of platforms and products. Both must have the same length.
        
    Returns
    -------
    full_lat, full_lon: tuple
        Two 2-tuples of the minimum and maximum latitude and longitude, respectively.
    min_max_dates: tuple
        A 2-tuple of the minimum and maximum time available to all products.
    """
    min_max_dates = np.empty((len(platforms), 2), dtype=object)
    min_max_lat = np.empty((len(platforms), 2))
    min_max_lon = np.empty((len(platforms), 2))
    for i, (platform, product) in enumerate(zip(platforms, products)):
        # Get the extents of the cube
        descriptor = api.get_query_metadata(platform=platform, product=product)

        # Save extents
        min_max_dates[i] = descriptor['time_extents']
        min_max_lat[i] = descriptor['lat_extents']
        min_max_lon[i] = descriptor['lon_extents']

    # Determine minimum and maximum longitudes that bound a common area among products
    min_lon = np.max(min_max_lon[:,0]) # The greatest minimum longitude among products
    max_lon = np.min(min_max_lon[:,1]) # The smallest maximum longitude among products
    min_lat = np.max(min_max_lat[:,0])
    max_lat = np.min(min_max_lat[:,1])
    full_lon = (min_lon, max_lon)
    full_lat = (min_lat, max_lat)
    return full_lat, full_lon, min_max_dates