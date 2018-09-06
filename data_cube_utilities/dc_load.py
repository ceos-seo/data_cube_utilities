import xarray as xr
from .clean_mask import landsat_qa_clean_mask, landsat_clean_mask_invalid
from xarray.ufuncs import logical_and as xr_and
from .sort import xarray_sortby_coord

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
    AssertionError: If no data is retrieved from any product.
    """
    datasets_temp = {} # Maps platforms to datasets to merge.
    clean_masks_temp = {} # Maps platforms to clean masks to merge.
    for product,platform in zip(products, platforms):
        current_load_params = dict(platform=platform, product=product)
        if isinstance(list(load_params.values())[0], dict): # Dict of dicts of platforms mapping to load params.
            current_load_params.update(load_params[platform])
        else:
            current_load_params.update(load_params)
        # Query the Data Cube
        dataset = dc.load(**current_load_params)
        if len(dataset.dims) > 0: # Sometimes data is not returned.
            # Get the clean mask for the appropriate LANDSAT satellite platform.
            current_masking_params = masking_params.get(platform, {})
            clean_mask_temp = landsat_qa_clean_mask(dataset, platform, **current_masking_params)
            # Get the mask for removing data ouside the accepted range of LANDSAT 7 and 8.
            clean_mask_temp = xr_and(clean_mask_temp, landsat_clean_mask_invalid(dataset))
            clean_masks_temp[platform] = clean_mask_temp
            # Mask the data.
            datasets_temp[platform] = dataset#.where(clean_mask_temp)
    assert len(datasets_temp) > 0, "No data was retrieved."
    # If mutliple non-empty datasets were retrieved, merge them and sort by time.
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