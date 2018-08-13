#Make a function for this since we will be doing this multiple times
from utils.data_cube_utilities.dc_mosaic import (ls7_unpack_qa, ls8_unpack_qa, ls5_unpack_qa)
import numpy as np

def create_2D_mosaic_clean_mask(clean_mask):
    """
    The clean mask of a mosaic should be determined by the compositing function (e.g. mean 
    mosaic, median mosaic, etc.). This is simply supposed to be a decent approximation of a 
    clean mask for a mosaic that has no time dimension.
    
    Parameters
    ----------
    clean_mask: np.ndarray
        The 3D clean mask used to construct the mosaic.
    
    Returns
    -------
    mosaic_clean_mask: np.ndarray
        A 2D clean mask for a mosaic.
    """
    mosaic_clean_mask = clean_mask[0]
    # Take the logical OR of clean masks through time.
    for i in range(1, clean_mask.shape[0]):
        mosaic_clean_mask = np.logical_or(mosaic_clean_mask, clean_mask[i])    
    return mosaic_clean_mask

def landsat_clean_mask_invalid(dataset):
    """
    Masks out invalid data according to the LANDSAT 7 and 8 surface reflectance 
    specifications. See this document: 
    https://landsat.usgs.gov/sites/default/files/documents/ledaps_product_guide.pdf pages 19-20.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray `Dataset` containing bands such as 'red', 'green', or 'blue'.
    """
    data_arr_names = [arr_name for arr_name in list(dataset.data_vars) if arr_name not in ['pixel_qa', 'radsat_qa', 'cloud_qa']]
    for data_arr_name in data_arr_names:
        dataset[data_arr_name] = dataset[data_arr_name].where((0 < dataset[data_arr_name]) & (dataset[data_arr_name] < 10000))
    return dataset
    

def landsat_qa_clean_mask(dataset, platform):
    """
    Returns a clean_mask for `dataset` that masks out clouds.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray (usually produced by `datacube.load()`) that contains a `pixel_qa` data 
        variable.
    platform: str
        A string denoting the platform to be used. Can be "LANDSAT_5", "LANDSAT_7", or 
        "LANDSAT_8".
        
    Returns
    -------
    clean_mask: numpy.ndarray
        A numpy array with the same number and order of dimensions as the coordinates of 
        `dataset`.
    """
    processing_options = {
        "LANDSAT_5": ls5_unpack_qa,
        "LANDSAT_7": ls7_unpack_qa,
        "LANDSAT_8": ls8_unpack_qa
    }
    
    #Clean mask creation to filter out pixels that are not suitable for analysis
    clear_xarray  = processing_options[platform](dataset.pixel_qa, "clear")  
    water_xarray  = processing_options[platform](dataset.pixel_qa, "water")
    
    #use logical or statement to elect viable pixels for analysis
    return np.logical_or(clear_xarray.values.astype(bool), water_xarray.values.astype(bool))

def xarray_values_in(data, values, data_vars=None):
    """
    Returns a mask for an xarray Dataset or DataArray, with `True` wherever the value is in values.
    
    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        The data to check for value matches.
    values: list-like
        The values to check for.
    data_vars: list-like
        The names of the data variables to check.
    
    Returns
    -------
    mask: np.ndarray
        A NumPy array shaped like ``data``. The mask can be used to mask ``data``.
        That is, ``data.where(mask)`` is an intended use.
    """
    if isinstance(data, xr.Dataset):
        mask = np.full_like(list(data.data_vars.values())[0], False, dtype=np.bool)
        for data_arr in data.data_vars.values():
            for value in values:
                mask = mask | (data_arr.values == value)
    elif isinstance(data, xr.DataArray):
        mask = np.full_like(data, False, dtype=np.bool)
        for value in values:
            mask = mask | (data.values == value)
    return mask
