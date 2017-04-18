import ccd
from dc_streamer import Streamer as _stream
import itertools as it
from functools import reduce
from datetime import datetime, timedelta

import warnings

import numpy as np
import xarray

############################################################################
## Auxilary Functions
############################################################################

###### Time FUNCTIONS #################################

def _n64_to_int(n64):
    b = n64.astype(object)
    return int(b/1000000000)  

def _n64_to_datetime(n64):
        b = n64.astype(object)
        return datetime.fromtimestamp(int(b/1000000000)) 
    
def _dt_to_sec(t):
    return datetime.fromtimestamp(t * 60 * 60 * 24)

    
###### Per Pixel FUNCTIONS ############################

def _run_ccd_on_pixel(ds):
    if 'time' not in ds.dims:
        raise Exception("You're missing time dims!")   

    available_bands = ds.data_vars
    scene_count = ds.dims['time']
    
    date  = [_n64_to_int(t)/60/60/24 for t in ds.time.values]

    red      = np.ones(scene_count)     if 'red'   not in available_bands else ds.red.values
    green    = np.ones(scene_count)     if 'green' not in available_bands else ds.green.values
    blue     = np.ones(scene_count)     if 'blue'  not in available_bands else ds.blue.values
    nir      = np.ones(scene_count)     if 'nir'   not in available_bands else ds.nir.values
    swir1    = np.ones(scene_count)     if 'swir1' not in available_bands else ds.swir1.values
    swir2    = np.ones(scene_count)     if 'swir2'  not in available_bands else ds.swir2.values
    
    thermals = np.ones(scene_count) * (273.15) *10  if 'thermal'  not in available_bands else ds.thermal.values
    qa = np.array(ds.cf_mask.values)
    
    params = (date,blue,green,red,nir,swir1, swir2, thermals, qa)
    
    return ccd.detect(*params) 

def _convert_ccd_results_into_dataset(results = None, model_dataset = None):
    
    start_times = [ datetime.fromtimestamp(model.start_day * 60 * 60 * 24) for model in results['change_models']]    
    
    intermediate_product = model_dataset.sel(time = start_times, method = 'nearest')
    
    new_dataset = xarray.DataArray(
            np.ones((intermediate_product.dims['time'],1,1)).astype(np.int16), 
            coords=[intermediate_product.time.values,
                [intermediate_product.latitude.values],
                [intermediate_product.longitude.values]],
        dims=['time', 'latitude', 'longitude']
        )
    
    return new_dataset.rename("continuous_change") 

def is_pixel(value):
        return (len(value.latitude.dims) == 0) and (len(value.longitude.dims) == 0)    
            
def clean_pixel( _ds, saturation_threshold = 10000):
    # Filter out over saturated values  
    ds = _ds
    mask =  (ds < saturation_threshold) & (ds >= 0)
    indices = [x for x,y in enumerate(mask.red.values) if y == True]
    return ds.isel(time = indices)


###### Visualization FUNCTIONS #########################
try:
	from matplotlib.pyplot import axvline
	import matplotlib.patches as patches
	from matplotlib import pyplot as plt
except:
	warnings.warn("Failed to load plotting library") 

def _lasso_eval(date=None, weights=None, bias=None):
    curves = [
        date,
        np.cos(2 * np.pi * (date / 365.25)),
        np.sin(2 * np.pi * (date / 365.25)),
        np.cos(4 * np.pi * (date / 365.25)),
        np.sin(4 * np.pi * (date / 365.25)),
        np.cos(6 * np.pi * (date / 365.25)),
        np.sin(6 * np.pi * (date / 365.25)),
    ]
    return np.dot(weights, curves) + bias

def intersect(a, b):
    return list(set(a) & set(b))

def _save_plot_to_file(plot = None, file = None, band_name = None):
    if isinstance(file_name, str):
        file_name = [file_name]
    for fn in file_name:
        plot.savefig(
            str.replace(fn, "$BAND$", band),
            orientation='landscape',
            papertype='letter',
            bbox_inches='tight')
        
def _plot_band(results= None,original_pixel= None, band=None, file_name = None):
        fig = plt.figure(1)
        fig.suptitle(band.title(), fontsize=18, verticalalignment='bottom')

        lastdt = None

        dateLabels = []

        for change_model in results["change_models"]:
            target = getattr(change_model, band)

            time = np.arange(change_model.start_day, change_model.end_day, 1)

            ax1 = fig.add_subplot(211)

            xy = [(t, _lasso_eval(
                date=t, weights=target.coefficients, bias=target.intercept))
                  for t in time]
            x, y = zip(*xy)
            x = [datetime.fromtimestamp(t * 60 * 60 * 24) for t in x]
            ax1.plot(x, y, label=target.coefficients)

            dt = datetime.fromtimestamp(change_model.start_day * 60 * 60 * 24)
            dateLabels.append(dt)

            if lastdt is not None:
                ax1.axvspan(lastdt, dt, color=(0, 0, 0, 0.1))

            dt = datetime.fromtimestamp(change_model.end_day * 60 * 60 * 24)
            dateLabels.append(dt)

            lastdt = dt

        if original_pixel is not None:  
            xy = [(_n64_to_datetime(x.time.values) + timedelta(0), x.values)
                  for x in clean_pixel(original_pixel)[band] if x < 5000]
            x, y = zip(*xy)
            ax2 = fig.add_subplot(211)
            ax2.scatter(x, y)

        ymin, ymax = ax1.get_ylim()
        for idx, dt in enumerate(dateLabels):
            plt.axvline(x=dt, linestyle='dotted', color=(0, 0, 0, 0.5))
            # Top, inside
            plt.text(
                dt,
                ymax,
                "\n" +  # HACK TO FIX SPACING
                dt.strftime('%b %d') + "  \n"  # HACK TO FIX SPACING
                ,
                rotation=90,
                horizontalalignment='right' if (idx % 2) else 'left',
                verticalalignment='top')

        plt.tight_layout()

        if file_name is not None:
            _save_plot_to_file(plot = plt, file = filename, band_name = band)
            
        plt.show()
        
        
###### STREAM FUNCTIONS ##########################################

def _generate_CCD_product(ds):
	try:
		return _convert_ccd_results_into_dataset(
			results = _run_ccd_on_pixel(ds),
			model_dataset = ds
		)
	except np.linalg.LinAlgError:
		# This is used to combat matrix inversion issues for Singular matrices.
		return None

def _dataset_to_pixel_chunker(_ds):
    lat_size = len(_ds.latitude)
    lon_size = len(_ds.longitude)
    cartesian = it.product(range(lat_size), range(lon_size))
    return _stream(cartesian).map(lambda x:_ds.isel(latitude = x[0], longitude = x[1]))
    
def _combine_pixels_to_form_dataset(stream):
    return reduce(lambda x,y : x.combine_first(y), stream)




###################################################################
## Callable Functions
###################################################################

def process_xarray(ds, distributed = False):
    change = None
    if distributed == False: 
        change = _stream([ds])\
            .flatmap(_dataset_to_pixel_chunker)\
            .map(_generate_CCD_product)\
            .reduce(_combine_pixels_to_form_dataset)
    else:
        change = _stream([ds])\
            .flatmap(_dataset_to_pixel_chunker)\
            .distributed_map(_generate_CCD_product)\
            .reduce(_combine_pixels_to_form_dataset)
                
    return (change.sum(dim = 'time') - 1).rename('change_volume')

def process_pixel(ds):
    if is_pixel(ds) is not True:
        raise Exception("Incorrect dimensions for pixel operation.")   

    duplicate_pixel = ds.copy(deep= True)
    ccd_results = _run_ccd_on_pixel(duplicate_pixel)  
    
    duplicate_pixel.attrs['ccd_results'] = ccd_results
    
    duplicate_pixel.attrs['ccd_start_times'] = [_dt_to_sec(model.start_day) for model in ccd_results['change_models']]
    duplicate_pixel.attrs['ccd_end_times'] =   [_dt_to_sec(model.end_day) for model in ccd_results['change_models']]
    duplicate_pixel.attrs['ccd_break_times'] = [_dt_to_sec(model.break_day) for model in ccd_results['change_models']]
    duplicate_pixel.attrs['ccd'] = True  
    return duplicate_pixel

def plot_pixel(ds, bands = None): 
    if 'ccd' not in list(ds.attrs.keys()):
        raise Exception("This pixel hasn't been processed by CCD. Use the `ccd.process_pixel()` function.")
        
    if bands is None or bands is []:
        possible_bands = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2','thermal']
        avaliable_bands = ds.data_vars
        bands = intersect(possible_bands, avaliable_bands)  
    
    for band in bands:
        _plot_band(results= ds.attrs['ccd_results'], original_pixel= ds, band=band)

        
