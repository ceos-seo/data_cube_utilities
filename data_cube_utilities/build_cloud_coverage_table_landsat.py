import datacube
from utils.data_cube_utilities.dc_mosaic import ls8_unpack_qa, ls7_unpack_qa 
import numpy as np
from functools import partial  
import pandas as pd

def build_cloud_coverage_table_landsat(product=None,
                                       platform=None,
                                       latitude=None,
                                       longitude=None,
                                       time=None,
                                       dc=None,
                                       extra_band='green'):
    
    if product   is None: raise Exception("product argument is required")
    if platform  is None: raise Exception("platform argument is required")
    if latitude  is None: raise Exception("latitude argument is required")
    if longitude is None: raise Exception("longitude argument is required")
 
    def clean_mask(ds, unpacking_func, bands):
        masks = [unpacking_func(ds, band) for band in bands]
        return np.logical_or(*masks).values
    
    unpack_function = {"LANDSAT_7": ls7_unpack_qa,
                       "LANDSAT_8": ls8_unpack_qa}

    dc = dc if dc else datacube.Datacube(app="")
    
    load_params = dict(product=product,
                       platform=platform,
                       latitude=latitude,
                       longitude=longitude,
                       measurements=[extra_band, 'pixel_qa'])
    
    if time is not None: 
        load_params["time"] = time
        
    geo_data = dc.load(**load_params)
    
    times = list(geo_data.time.values)
    scene_slice_list = list(map(lambda t: geo_data.sel(time=str(t)), times))
    
    def create_clean_mask_list(ds):
        return clean_mask(ds, unpacking_func=unpack_function[platform], bands=["clear", "water"])
    clean_mask_list = list(map(lambda ds: create_clean_mask_list(ds.pixel_qa), scene_slice_list))
    no_data_mask_list = list(map(lambda ds: (ds[extra_band] == -9999).values, scene_slice_list))
    # This method of creating `percentage_list` gives the percentage of pixels with data (i.e. not no_data (-9999))
    # which are also not cloud.
#     def create_clean_percentage_with_data_list(masks_tup):
#         """Merges two masks - passed as a tuple - and calculates the percentage
#            of pixels that are clean and have data."""
#         clean_mask, no_data_mask = masks_tup
#         merged_mask = clean_mask & ~no_data_mask
#         return merged_mask.sum() / (~no_data_mask).sum() * 100
#     percentage_list = list(map(create_clean_percentage_with_data_list, 
#                                zip(clean_mask_list, no_data_mask_list)))
    # This method of creating `percentage_list` gieves the percentage of all pixels 
    # which are also not cloud.
    percentage_list = [clean_mask.mean()*100 for clean_mask in clean_mask_list]
    clean_pixel_count_list = list(map(np.sum, clean_mask_list))
    
    data = {"times": times,
            # "clean_percentage" is the percent of pixels that are not no_data which are clear.
            "clean_percentage": percentage_list,
            "clean_count": clean_pixel_count_list }
    
    return geo_data, pd.DataFrame(data=data, columns=["times", "clean_percentage", "clean_count"])
