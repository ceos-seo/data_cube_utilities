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
    specifications.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray `Dataset` containing bands such as 'red', 'green', or 'blue'.
    """
    data_bands = dataset.drop('pixel_qa')
    return data_bands.where((0 < data_bands) & (data_bands < 10000))
    

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



