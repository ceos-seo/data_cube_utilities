import numpy as np
import xarray as xr
from .clean_mask import landsat_qa_clean_mask, landsat_clean_mask_invalid
from xarray.ufuncs import logical_and as xr_and
from .sort import xarray_sortby_coord
from .aggregate import xr_scale_res

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

def merge_datasets(datasets_temp, clean_masks_temp, masks_per_platform=None,
                   x_coord='longitude', y_coord='latitude'):
    """
    Merges dictionaries of platforms mapping to datasets, dataset clean masks,
    and lists of other masks into one dataset, one dataset clean mask, and one
    of each type of other mask, ordering all by time.

    Parameters
    ----------
    datasets_temp, clean_masks_temp, masks_per_platform: dict
        Dictionaries that map, respectively, platforms to `xarray.Dataset`
        or `xarray.DataArray` objects to merge, `xarray.DataArray` masks
        to merge, and lists of `xarray.DataArray` masks to merge separately.
        All entries must have a 'time' dimension.
    x_coord, y_coord: str
        Names of the x and y coordinates in the datasets in `datasets_temp`.

    Returns
    -------
    dataset: xarray.Dataset or xarray.DataArray
        The raw data requested. Can be cleaned with `dataset.where(clean_mask)`.
    clean_mask: xarray.DataArray
        The clean mask.
    masks: list of xarray.DataArray
        A list of individual masks.

    Raises
    ------
    AssertionError: If no data was retrieved for any query
                    (i.e. `len(datasets_temp) == 0`).
    
    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    def xr_set_same_coords(datasets):
        first_ds = datasets[0]
        for i, ds in enumerate(datasets):
            datasets[i] = \
                ds.assign_coords(**{x_coord:first_ds[x_coord],
                                    y_coord:first_ds[y_coord]})

    assert len(datasets_temp) > 0, "No data was retrieved." # No data for any query.
    # If multiple non-empty datasets were retrieved, merge them and sort by time.
    masks = None
    if len(datasets_temp) > 1:
        # Merge datasets.
        datasets_temp_list = list(datasets_temp.values())
        # Set same x and y coords so `xr.concat()` concatenates as intended.
        xr_set_same_coords(datasets_temp_list)
        dataset = xr.concat(datasets_temp_list, dim='time')
        dataset = xarray_sortby_coord(dataset, 'time')
        # Merge clean masks.
        clean_masks_temp_list = list(clean_masks_temp.values())
        xr_set_same_coords(clean_masks_temp_list)
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

def load_simple(dc, platform, product, frac_res=None, abs_res=None,
                load_params={}, masking_params={}, indiv_masks=None):
    """
    Simplifies loading from the Data Cube by retrieving a dataset along
    with its mask.

    Parameters
    ----------
    dc: datacube.api.core.Datacube
        The Datacube instance to load data with.
    platform, product: str
        Strings denoting the platform and product to retrieve data for.
    frac_res: float
        The fraction of the original resolution to scale to. Must be postive.
        Note that this can be greater than 1.0, in which case the resolution
        is upsampled.
    abs_res: list-like
        A list-like of the number of pixels for the x and y axes, respectively.
        Overrides `frac_res` if specified.
    load_params: dict, optional
        A dictionary of parameters for `dc.load()`.
        Here are some common load parameters:
        *lat, lon: list-like 2-tuples of minimum and maximum values for
                             latitude and longitude, respectively.*
        *time: list-like     A 2-tuple of the minimum and maximum times
                             for acquisitions.*
        *measurements: list-like The list of measurements to retrieve
                                 from the Datacube.*
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
    # Scale resolution if specified.
    if frac_res is not None or abs_res is not None:
        dataset = xr_scale_res(dataset, frac_res=frac_res, abs_res=abs_res)
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

def load_multiplatform(dc, platforms, products, frac_res=None, abs_res=None,
                       load_params={}, masking_params={}, indiv_masks=None):
    """
    Load and merge data as well as clean masks given a list of platforms
    and products. Currently only tested on Landsat data.
    
    Parameters
    ----------
    dc: datacube.api.core.Datacube
        The Datacube instance to load data with.
    platforms, products: list-like
        A list-like of platforms and products. Both must have the same length.
    frac_res: float
        The fraction of the original resolution to scale to. Must be postive.
        Note that this can be greater than 1.0, in which case the resolution
        is upsampled. The resolution used for all products will be the
        minimum resolution for latitude and longitude among any of them.
    abs_res: list-like
        A list-like of the number of pixels for the x and y axes, respectively.
        Overrides `frac_res` if specified.
    load_params: dict, optional
        A dictionary of parameters for `dc.load()` or a dictionary of
        dictionaries of such parameters, mapping platform names to parameter
        dictionaries (primarily useful for selecting different time ranges).
        Here are some common load parameters:
        *lat, lon: list-like 2-tuples of minimum and maximum values for
                             latitude and longitude, respectively.*
        *time: list-like     A pair of the minimum and maximum times
                             for acquisitions or a list of such pairs.*
        *measurements: list-like The list of measurements to retrieve from
                                 the Datacube.*
        For example, to load data with different time ranges for different
        platforms, we could pass the following:
        `{'LANDSAT_7': dict(**common_load_params, time=ls7_date_range),
          'LANDSAT_8': dict(**common_load_params, time=ls8_date_range)}`,
          where `common_load_params` is a dictionary of load parameters common
          to both - most notably 'lat', 'lon', and 'measurements' - and the
          'date_range' variables are .
    masking_params: dict, optional
        A dictionary mapping platform names to a dictionary of keyword
        arguments for corresponding masking functions.
        For example: {'LANDSAT_7': {'cover_types':['cloud']},
                      'LANDSAT_8': {'firstcover_types': ['cloud']}}
        would retain only clouds, because `landsat_qa_clean_mask()`  is used
        to create clean masks for the Landsat family of platforms.
    indiv_masks: list
        A list of masks to return (e.g. ['water']). These do not have to be
        the same used to create the returned clean mask.
    
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
    # Determine what resolution the data will be scaled to.
    if frac_res is not None and abs_res is None:
        prod_info = dc.list_products()
        resolutions = prod_info[prod_info['name'].isin(products)]\
                               ['resolution'].values
        # Determine the minimum resolution, which is actually the maximum
        # value resolution, since resolution is measured in degrees per pixel.
        # The first resolution is for latitude (y) and is negative.
        # The second resolution is for longitude (x) and is positive.
        min_res = [0]*2
        for res in resolutions:
            min_res[0] = res[0] if res[0] < min_res[0] else min_res[0]
            min_res[1] = res[1] if min_res[1] < res[1] else min_res[1]
        # Take reciprocal to convert pixels per degree to degrees per pixel.
        # Reverse to be in order (x, y).
        min_res = [abs(frac_res*(1/res)) for res in min_res][::-1]

        # Calculate the absolute resolution.
        x, y = load_params.get('lon', None), load_params.get('lat', None)
        x, y = load_params.get('longitude', x), load_params.get('latitude', y)
        x_y_rng = abs(x[1] - x[0]), abs(y[1] - y[0])
        abs_res = [round(res*rng) for res, rng in zip(min_res, x_y_rng)]

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
                        load_simple(dc, platform, product, abs_res=abs_res,
                                    load_params=time_range_load_params,
                                    masking_params=masking_params,
                                    indiv_masks=indiv_masks)
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
                    load_simple(dc, platform, product, abs_res=abs_res,
                                load_params=current_load_params,
                                masking_params=masking_params,
                                indiv_masks=indiv_masks)
                if indiv_masks is not None:
                    masks_per_platform[platform] = masks
            except (AssertionError):
                continue
    return merge_datasets(datasets_temp, clean_masks_temp, masks_per_platform)

def get_product_extents(api, platform, product):
    """
    Returns the minimum and maximum latitude, longitude, and date range of a product.

    Parameters
    ----------
    api: DataAccessApi
        An instance of `DataAccessApi` to get query metadata from.
    platform, product: str
        Names of the platform and product to query extent information for.

    Returns
    -------
    full_lat, full_lon: tuple
        Two 2-tuples of the minimum and maximum latitude and longitude, respectively.
    min_max_dates: tuple of datetime.datetime
        A 2-tuple of the minimum and maximum time available.
    """
    # Get the extents of the cube
    descriptor = api.get_query_metadata(platform=platform, product=product, measurements=[])
    min_max_lat = descriptor['lat_extents']
    min_max_lon = descriptor['lon_extents']
    min_max_dates = descriptor['time_extents']
    return min_max_lat, min_max_lon, min_max_dates

def get_overlapping_area(api, platforms, products):
    """
    Returns the minimum and maximum latitude, longitude, and date range of the overlapping
    area for a set of products.
    
    Parameters
    ----------
    api: DataAccessApi
        An instance of `DataAccessApi` to get query metadata from.
    platforms, products: list-like of str
        A list-like of names of platforms and products to query extent information for.
        These lists must have the same length.
        
    Returns
    -------
    full_lat, full_lon: tuple
        Two 2-tuples of the minimum and maximum latitude and longitude, respectively.
    min_max_dates: numpy.ndarray of datetime.datetime
        A 2D NumPy array with shape (len(products), 2), in which rows contain the minimum
        and maximum time available for corresponding products.
    """
    min_max_dates = np.empty((len(platforms), 2), dtype=object)
    min_max_lat = np.empty((len(platforms), 2))
    min_max_lon = np.empty((len(platforms), 2))
    for i, (platform, product) in enumerate(zip(platforms, products)):
        min_max_lat[i], min_max_lon[i], min_max_dates[i] = \
            get_product_extents(api, platform, product)
    # Determine minimum and maximum lat and lon extents that bound a common area among the
    # products, which are the greatest minimums and smallest maximums.
    min_lon, max_lon = np.max(min_max_lon[:,0]), np.min(min_max_lon[:,1])
    min_lat, max_lat = np.max(min_max_lat[:,0]), np.min(min_max_lat[:,1])
    full_lon = (min_lon, max_lon)
    full_lat = (min_lat, max_lat)
    return full_lat, full_lon, min_max_dates