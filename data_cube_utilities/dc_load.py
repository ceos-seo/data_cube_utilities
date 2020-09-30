import itertools
import numpy as np
import xarray as xr
from xarray.ufuncs import logical_and as xr_and


## Misc ##

def is_dataset_empty(ds:xr.Dataset) -> bool:
    checks_for_empty = [
                        lambda x: len(x.dims) == 0,      # Dataset has no dimensions
                        lambda x: len(x.data_vars) == 0, # Dataset has no variables
                        lambda x: list(x.data_vars.values())[0].count().values == 0 # Data variables are empty
                       ]
    for f in checks_for_empty:
        if f(ds) == True:
            return True
    return False

## End Misc ##

## Combining Data ##

def match_prods_res(dc, products, method='min'):
    """
    Determines a resolution that matches a set of Data Cube products -
    either the minimum or maximum resolution along the x and y dimensions.
    Product resolutions are derived from Data Cube metadata for those products.

    Parameters
    ----------
    dc: datacube.Datacube
        A connection to the Data Cube to determine the resolution of
        individual products from.
    products: list of str
        The names of the products to find a matching resolution for.
    method: str
        The method of finding a matching resolution. The options are
        ['min', 'max'], which separately determine the y and x resolutions
        as the minimum or maximum among all selected products.

    Returns
    -------
    res: list
        A list of the y and x resolutions, in that order.
    """
    if method not in ['min', 'max']:
        raise ValueError("The method \"{}\" is not supported. "
                         "Please choose one of ['min', 'max'].".format(method))
    prod_info = dc.list_products()
    resolutions = prod_info[prod_info['name'].isin(products)] \
        ['resolution'].values

    # The first resolution is for y and is negative.
    # The second resolution is for x and is positive.
    if method == 'min':
        # Determine the minimum resolution, which is actually the maximum
        # value resolution, since resolution is measured in degrees per pixel.
        matching_res = [0] * 2
        for res in resolutions:
            matching_res[0] = res[0] if res[0] < matching_res[0] else matching_res[0]
            matching_res[1] = res[1] if matching_res[1] < res[1] else matching_res[1]
    else:
        matching_res = [-np.inf, np.inf]
        for res in resolutions:
            matching_res[0] = res[0] if matching_res[0] < res[0] else matching_res[0]
            matching_res[1] = res[1] if res[1] < matching_res[1] else matching_res[1]
    return matching_res


def match_dim_sizes(
    dc, products, x, y, 
    x_y_coords=['longitude', 'latitude'], method='min',
    load_kwargs=None):
    """
    Returns the x and y dimension sizes that match some x and y extents for some products.
    This is useful when determining an absolute resolution to scale products to with
    `xr_scale_res()` in the `aggregate.py` utility file.

    Parameters
    ----------
    dc: datacube.Datacube
        A connection to the Data Cube to determine the resolution of
        individual products from.
    products: list of str
        The names of the products to find a matching resolution for.
    x: list-like
        A list-like of the minimum and maximum x-axis (e.g. longitude) extents for the products.
    y: list-like
        A list-like of the minimum and maximum y-axis (e.g. latitude) extents for the products.
    x_y_coords: list-like or dict
        Either a list-like of the x and y coordinate names or a dictionary mapping product names
        to such list-likes.
    method: str
        The method of finding a matching resolution. The options are
        ['min', 'max'], which separately determine the y and x resolutions
        as the minimum or maximum among all selected products.
    load_kwargs: dict
        Dictionary of product names mapping to parameter dictionaries
        passed to `datacube.Datacube.load()` for the corresponding products.

    Returns
    -------
    abs_res: list
        A list of desired y and x dimension sizes, in that order.
    same_dim_sizes: bool
        Whether all of the dimension sizes were the same.
    """
    coords = []
    if isinstance(x_y_coords, dict):
        for product in products:
            coords.append(x_y_coords[product])
    else:
        coords = [x_y_coords] * len(products)

    load_params = {product:dict(product=product, x=x, y=y, measurements=[]) for product in products}
    if load_kwargs is not None:
        for product in products:
            load_params[product].update(**load_kwargs.get(product, {}) if load_kwargs is not None else {})

    datasets_empty = [dc.load(product=product, x=x, y=y, measurements=[]) for product in products]

    # First check if all datasets will load with the same x and y dimension sizes.
    same_dim_sizes = True
    first_dataset_dim_size = [datasets_empty[0][coords[0][0]].size, datasets_empty[0][coords[0][1]].size]
    for i in range(1, len(datasets_empty)):
        if first_dataset_dim_size != [datasets_empty[i][coords[i][0]].size, datasets_empty[i][coords[i][1]].size]:
            same_dim_sizes = False
            break

    if method == 'min':
        abs_res = [np.inf, np.inf]
        for i in range(len(datasets_empty)):
            res = [datasets_empty[i][coords[i][0]].size, datasets_empty[i][coords[i][1]].size]
            abs_res[0] = res[0] if res[0] < abs_res[0] else abs_res[0]
            abs_res[1] = res[1] if res[1] < abs_res[1] else abs_res[1]
    else:
        abs_res = [0] * 2
        for i in range(len(datasets_empty)):
            res = [datasets_empty[i][coords[i][0]].size, datasets_empty[i][coords[i][1]].size]
            abs_res[0] = res[0] if abs_res[0] < res[0] else abs_res[0]
            abs_res[1] = res[1] if abs_res[1] < res[1] else abs_res[1]

    return abs_res, same_dim_sizes

def xarray_concat_and_merge(*args, concat_dim='time', sort_dim='time'):
    """
    Given parameters that are each a list of `xarray.Dataset` objects, merge each list 
    into an `xarray.Dataset` object and return all such objects in the same order.

    Parameters
    ----------
    *args: list of lists of `xarray.Dataset`.
        A list of lists of `xarray.Dataset` objects to merge.
    concat_dim, sort_dim: str
        The string name of the dimension to concatenate or sort by the data.

    Returns
    -------
    merged: list of `xarray.Dataset`
        A tuple of the same length as `*args`, containing the merged data. 
    """
    from .sort import xarray_sortby_coord

    merged = []
    for i, arg in enumerate(args):
        dataset_temp = xr.concat(arg, dim=concat_dim)
        merged.append(xarray_sortby_coord(dataset_temp, coord=sort_dim))
    return merged

def merge_datasets(datasets_temp, clean_masks_temp, masks_per_platform=None,
                   x_coord='longitude', y_coord='latitude'):
    """
    Merges dictionaries of platform names mapping to datasets, dataset clean masks,
    and lists of other masks into one dataset, one dataset clean mask, and one
    of each type of other mask, ordering all by time.

    Parameters
    ----------
    datasets_temp: dict
        Dictionary that maps platforms to `xarray.Dataset` or `xarray.DataArray`
        objects to merge to make the output `dataset`.
        Must have a 'time' dimension.
    clean_masks_temp: dict
        Dictionary that maps platforms to `xarray.DataArray` masks to merge to make the output `clean_mask`.
        Must have a 'time' dimension.
    masks_per_platform: dict
        Dictionary that maps platforms to `xarray.DataArray` masks to merge to make the output `masks`.
        Must have a 'time' dimension.
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
    """
    from .sort import xarray_sortby_coord
    from .aggregate import xr_scale_res

    def xr_set_same_coords(datasets):
        first_ds = datasets[0]
        for i, ds in enumerate(datasets):
            datasets[i] = \
                ds.assign_coords(**{x_coord: first_ds[x_coord],
                                    y_coord: first_ds[y_coord]})

    masks = None
    if len(datasets_temp) == 0:  # No data was retrieved.
        return xr.Dataset(), xr.DataArray(np.array(None)), np.array(None) if masks_per_platform is not None else None
    elif len(datasets_temp) == 1:  # Select the only dataset.
        dataset = datasets_temp[list(datasets_temp.keys())[0]]
        clean_mask = clean_masks_temp[list(clean_masks_temp.keys())[0]]
        if masks_per_platform is not None:
            masks = masks_per_platform[list(masks_per_platform.keys())[0]]
    else:  # Merge datasets.
        # Make sure all datasets have the same sizes in the x and y dimensions.
        datasets_temp_list = list(datasets_temp.values())
        max_num_x = max([len(dataset[x_coord]) for dataset in datasets_temp_list])
        max_num_y = max([len(dataset[y_coord]) for dataset in datasets_temp_list])
        datasets_temp_list = [xr_scale_res(dataset, x_coord=x_coord, y_coord=y_coord,
                                           abs_res=(max_num_x, max_num_y))
                              for dataset in datasets_temp_list]
        # Set same x and y coords so `xr.concat()` concatenates as intended.
        xr_set_same_coords(datasets_temp_list)
        dataset = xr.concat(datasets_temp_list, dim='time')
        dataset = xarray_sortby_coord(dataset, 'time')

        # Merge clean masks.
        # Make sure all clean masks have the same sizes in the x and y dimensions.
        clean_masks_temp_list = list(clean_masks_temp.values())
        clean_masks_temp_list = [xr_scale_res(clean_mask.astype(np.int8), x_coord=x_coord, y_coord=y_coord,
                                              abs_res=(max_num_x, max_num_y))
                                 for clean_mask in clean_masks_temp_list]
        # Set same x and y coords so `xr.concat()` concatenates as intended.
        xr_set_same_coords(clean_masks_temp_list)
        clean_mask = xr.concat(clean_masks_temp_list, dim='time')
        clean_mask = xarray_sortby_coord(clean_mask, 'time').astype(np.bool)
        # Merge masks.
        if masks_per_platform is not None:
            num_platforms = len(masks_per_platform.keys())
            num_masks = len(list(masks_per_platform.values())[0])
            np_platform_masks = np.empty((num_platforms, num_masks), dtype=object)
            for i, mask_list in enumerate(masks_per_platform.values()):
                np_platform_masks[i] = mask_list
            masks = []
            for j in range(num_masks):
                masks.append(xr.concat(list(np_platform_masks[:, j]), dim='time'))
    return dataset, clean_mask, masks

## End Combining Data ##

## Extents ##

def get_product_extents(api, platform, product, **kwargs):
    """
    Returns the minimum and maximum latitude, longitude, and date range of a product.

    Parameters
    ----------
    api: DataAccessApi
        An instance of `DataAccessApi` to get query metadata from.
    platform, product: str
        Names of the platform and product to query extent information for.
    **kwargs: dict
        Keyword arguments for `api.get_query_metadata()`.

    Returns
    -------
    full_lat, full_lon: tuple
        Two 2-tuples of the minimum and maximum latitude and longitude, respectively.
    min_max_dates: tuple of datetime.datetime
        A 2-tuple of the minimum and maximum time available.
    """
    # Get the extents of the cube
    descriptor = api.get_query_metadata(platform=platform, product=product, **kwargs)
    min_max_lat = descriptor['lat_extents']
    min_max_lon = descriptor['lon_extents']
    min_max_dates = descriptor['time_extents']
    return min_max_lat, min_max_lon, min_max_dates

def get_overlapping_area(api, platforms, products, **product_kwargs):
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
    **product_kwargs: dict
        A dictionary mapping product names to keyword arguments for
        `get_product_extents()`
        
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
            get_product_extents(api, platform, product,
                                **product_kwargs.get(product, dict()))
    # Determine minimum and maximum lat and lon extents that bound a common area among the
    # products, which are the greatest minimums and smallest maximums.
    min_lon, max_lon = np.max(min_max_lon[:,0]), np.min(min_max_lon[:,1])
    min_lat, max_lat = np.max(min_max_lat[:,0]), np.min(min_max_lat[:,1])
    full_lon = (min_lon, max_lon)
    full_lat = (min_lat, max_lat)
    return full_lat, full_lon, min_max_dates

## End Extents ##

## Undesired Acquisition Removal ##

def find_desired_acq_inds(dataset=None, clean_mask=None, time_dim='time', pct_clean=None, not_empty=False):
    """
    Returns indices of acquisitions that meet a specified set of criteria in
    an `xarray.Dataset` or `xarray.DataArray`.

    Parameters
    ----------
    dataset: xarray.Dataset or xarray.DataArray
        The `xarray` object to remove undesired acquisitions from.
    clean_mask: xarray.DataArray
        A boolean `xarray.DataArray` denoting the "clean" values in `dataset`.
        More generally, in this mask, `True` values are considered desirable.
    time_dim: str
        The string name of the time dimension.
    pct_clean: float
        The minimum percent of "clean" (or "desired") pixels required to keep an acquisition.
        Requires `clean_mask` to be supplied.
    not_empty: bool
        Whether to remove empty acquisitions or not.
        Here, an empty acquisition is one that contains all NaN values.
        Requires `dataset` to be supplied.

    Returns
    -------
    acq_inds_to_keep: list of int
        A list of indices of acquisitions that meet the specified criteria.
    """
    if pct_clean is not None:
        assert clean_mask is not None, "If `pct_clean` is supplied, then `clean_mask` must also be supplied."
    if not_empty:
        assert dataset is not None, "If `not_empty==True`, then `dataset` must be supplied."
    acq_inds_to_keep = []
    for time_ind in range(len(dataset[time_dim])):
        remove_acq = False
        if pct_clean is not None:
            acq_pct_clean = clean_mask.isel(time=time_ind).mean()
            remove_acq = acq_pct_clean < pct_clean
        if not_empty:
            remove_acq = is_dataset_empty(dataset.isel(time=time_ind))
        if not remove_acq:
            acq_inds_to_keep.append(time_ind)
    return acq_inds_to_keep


def group_dates_by_day(dates):
    """
    Given a list of dates, return the list of lists of dates grouped by day.

    Parameters
    ----------
    dates: List[np.datetime64]

    Returns
    -------
    grouped_dates: List[List[np.datetime64]]
    """
    generate_key = lambda b: ((b - np.datetime64('1970-01-01T00:00:00Z')) / (np.timedelta64(1, 'h') * 24)).astype(int)
    return [list(group) for key, group in itertools.groupby(dates, key=generate_key)]


def reduce_on_day(ds, reduction_func=np.nanmean):
    """
    Combine data in an `xarray.Dataset` for dates with the same day

    Parameters
    ----------
    ds: xr.Dataset
    reduction_func: np.ufunc

    Returns
    -------
    reduced_ds: xr.Dataset
    """
    from .dc_mosaic import restore_or_convert_dtypes

    # Save dtypes to convert back to them.
    dataset_in_dtypes = {}
    for band in ds.data_vars:
        dataset_in_dtypes[band] = ds[band].dtype

    # Group dates by day into date_groups
    day_groups = group_dates_by_day(ds.time.values)

    # slice large dataset into many smaller datasets by date_group
    group_chunks = (ds.sel(time=t) for t in day_groups)

    # reduce each dataset using something like "average" or "median" such that many values for a day become one value
    group_slices = (_ds.reduce(reduction_func, dim="time") for _ds in group_chunks if "time" in dict(ds.dims).keys())

    # recombine slices into larger dataset
    new_dataset = xr.concat(group_slices, dim="time")

    # rename times values using the first time in each date_group
    new_times = [day_group[0] for day_group in day_groups]  # list(map(get_first, day_groups))
    new_dataset = new_dataset.reindex(dict(time=np.array(new_times)))

    restore_or_convert_dtypes(None, None, dataset_in_dtypes, new_dataset)

    return new_dataset

## End Undesired Acquisition Removal ##