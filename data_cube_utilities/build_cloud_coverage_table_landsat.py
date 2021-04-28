from functools import partial  
import datacube
from .clean_mask import landsat_clean_mask_invalid, landsat_qa_clean_mask
import numpy as np
import xarray as xr
import pandas as pd


def build_cloud_coverage_table_landsat(product,
                                       platform,
                                       collection,
                                       level,
                                       latitude,
                                       longitude,
                                       time     = None,
                                       dc       = None,
                                       extra_band = 'green',
                                       extra_load_params = {}):
    dc = dc if dc is not None else datacube.Datacube(app = "")
    
    load_params = dict(product=product,
                       latitude = latitude,
                       longitude = longitude,
                       measurements = [extra_band, 'pixel_qa'],
                       **extra_load_params)
    
    if time is not None: 
        load_params["time"] = time
    
    landsat_dataset = dc.load(**load_params).persist()
    clean_mask = landsat_qa_clean_mask(landsat_dataset, platform=platform, 
                                       collection=collection, level=level) & \
                 landsat_clean_mask_invalid(landsat_dataset, platform, collection, level)
    
    data_mask = xr.full_like(clean_mask, True) 
    band_no_data_values = dc.list_measurements().loc[product, 'nodata']
    if band_no_data_values is not None:
        for data_var_name in landsat_dataset.data_vars:
            band_data_mask = landsat_dataset[data_var_name] != band_no_data_values[data_var_name]
            data_mask = data_mask & band_data_mask
    clean_data_mask = clean_mask & data_mask
    
    landsat_dataset = landsat_dataset.where(clean_data_mask)
    
    times = list(landsat_dataset.time.values)
    scene_slice_list = list(map(lambda t: landsat_dataset.sel(time = str(t)), times))
    
    clean_data_mask_list = [clean_data_mask.sel(time=str(time)).values for time in clean_data_mask.time.values]
    # Calculate the percentage of all pixels which are not cloud.
    percentage_list = [clean_data_mask.mean()*100 for clean_data_mask in clean_data_mask_list]
    clean_pixel_count_list = list(map(np.sum, clean_data_mask_list))
    
    data = {"times": times,
            "clean_percentage": percentage_list,
            "clean_count": clean_pixel_count_list }
    
    return landsat_dataset, pd.DataFrame(data=data, columns=["times", "clean_percentage", "clean_count"]), \
           clean_mask, data_mask, clean_data_mask