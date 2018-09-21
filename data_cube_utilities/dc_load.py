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
        current_concat_dim = concat_dim[i] if isinstance (concat_dim, list) else concat_dim
        current_sort_dim = sort_dim[i] if isinstance (sort_dim, list) else sort_dim
        dataset_temp = xr.concat(arg, dim=concat_dim)
        merged.append(xarray_sortby_coord(dataset_temp, coord=sort_dim))
    return merged

def merge_datasets(datasets_temp, clean_masks_temp):
    """
    Merges dictionaries of platforms mapping to datasets and 
    dataset masks into one dataset and dataset mask.

    Parameters
    ----------
    datasets_temp, clean_masks_temp: dict
        Maps platforms to datasets to merge and dataset masks to merge.

    Returns
    -------
    dataset: xarray.Dataset
        The data requested.
    clean_mask: xarray.DataArray
        The clean mask, formed as a logical AND of all masks used.

    Raises
    ------
    AssertionError: If no data was retrieved for any query (i.e. `len(datasets_temp) == 0`).
    """
    assert len(datasets_temp) > 0, "No data was retrieved." # No data for any query.
    # If multiple non-empty datasets were retrieved, merge them and sort by time.
    if len(datasets_temp) > 1:
        # Merge datasets.
        datasets_temp_list = list(datasets_temp.values())
        dataset = xr.concat(datasets_temp_list, dim='time')
        dataset = xarray_sortby_coord(dataset, 'time')
        # Merge clean masks.
        clean_masks_temp_list = list(clean_masks_temp.values())
        clean_mask = xr.concat(clean_masks_temp_list, dim='time')
        clean_mask = xarray_sortby_coord(clean_mask, 'time')
    else: # Select the only dataset.
        dataset = datasets_temp[list(datasets_temp.keys())[0]]
        clean_mask = clean_masks_temp[list(clean_masks_temp.keys())[0]]
    return dataset, clean_mask

def load_simple(dc, platform, product, load_params={}, masking_params={}):
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
        A dictionary mapping platform names to a dictionary of keyword arguments for corresponding masking functions.
        For example: {'LANDSAT_7': {'cover_types':['cloud']}, 'LANDSAT_8': {'cover_types': ['cloud']}} would retain
        only clouds, because `landsat_qa_clean_mask()` is used for the Landsat family of platforms.

    Returns
    -------
    dataset: xarray.Dataset
        The data requested.
    clean_mask: xarray.DataArray
        The clean mask, formed as a logical AND of all masks used.

    Raises
    ------
    AssertionError: If no data is retrieved for this query.
    """
    current_load_params = dict(platform=platform, product=product)
    current_masking_params = masking_params.get(platform, {})
    current_load_params.update(load_params)
    dataset = dc.load(**current_load_params)
    assert len(dataset.dims) > 0, "No data was retrieved." # No data for this query.
    # Get the clean mask for the appropriate LANDSAT satellite platform.
    clean_mask = landsat_qa_clean_mask(dataset, platform, **current_masking_params)
    # Get the mask for removing data ouside the accepted range of LANDSAT 7 and 8.
    clean_mask = xr_and(clean_mask, landsat_clean_mask_invalid(dataset))
    return dataset, clean_mask

def load_multiplatform(dc, platforms, products, load_params={}, masking_params={}):
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
    
    Returns
    -------
    dataset: xarray.Dataset
        The data requested.
    clean_mask: xarray.DataArray
        The clean mask, formed as a logical AND of all masks used.
        
    Raises
    ------
    AssertionError: If no data is retrieved from any product.
    """
    datasets_temp = {} # Maps platforms to datasets to merge.
    clean_masks_temp = {} # Maps platforms to clean masks to merge.
    for product,platform in zip(products, platforms):
        current_load_params = dict(platform=platform, product=product)
        current_masking_params = masking_params.get(platform, {})
        
        # Handle `load_params` as a dict of dicts of platforms mapping to load params.
        if isinstance(list(load_params.values())[0], dict): 
            current_load_params.update(load_params.get(platform, {}))
        else: # Handle `load_params` as a dict of load params.
            current_load_params.update(load_params)
        
        # Load each time range of data.
        time = current_load_params.get('time')
        if isinstance(time[0], list): # Handle `time` as a list of time ranges.
            datasets_time_parts = []
            clean_masks_time_parts = []
            for time_range in time:
                time_range_load_params = current_load_params
                time_range_load_params['time'] = time_range
                try:
                    dataset_time_part, clean_mask_time_part = \
                        load_simple(dc, platform, product, time_range_load_params, masking_params)
                    datasets_time_parts.append(dataset_time_part)
                    clean_masks_time_parts.append(clean_mask_time_part)    
                except (AssertionError):
                    continue
            datasets_temp[platform], clean_masks_temp[platform] = \
                xarray_concat_and_merge(datasets_time_parts, clean_masks_time_parts)
        else: # Handle `time` as a single time range.
            try:
                datasets_temp[platform], clean_masks_temp[platform] = \
                    load_simple(dc, platform, product, load_params, masking_params)
            except (AssertionError):
                continue
    return merge_datasets(datasets_temp, clean_masks_temp)

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
    return full_lat, full_lon