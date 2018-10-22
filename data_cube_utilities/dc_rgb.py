import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from time import time
import numpy as np


# Change the bands (RGB) here if you want other false color combinations
def rgb(dataset, at_index = 0, bands = ['red', 'green', 'blue'], paint_on_mask = [],
        min_possible = 0, max_possible = 10000, width = 10):
    """
    Creates a figure showing an area, using three specified bands as the rgb componenets.

    Parameters
    ----------
    dataset: xarray.Dataset
        A Dataset containing latitude and longitude coordinates, in that order.
        Must contain the data variables specified in the `bands` parameter.
    bands: list-like
        A list-like containing 3 names of data variables in `dataset` to use as the red, green, and blue
        bands, respectively.
    min_possible: int
        The minimum valid value for relevant bands according to the platform used to retrieve the data in `dataset`.
        For example, for Landsat this is generally 0.
    max_possible: int
        The maximum valid value for relevant bands according to the platform used to retrieve the data in `dataset`.
        For example, for Landsat this is generally 10000.
    width: int
        The width of the figure in inches.
    """
    def aspect_ratio_helper(x,y, fixed_width = 20):
        width = fixed_width
        height = y * (fixed_width / x)
        return (width, height)
    
    ### < Dataset to RGB Format, needs float values between 0-1 
    rgb = np.stack([dataset[bands[0]],
                    dataset[bands[1]],
                    dataset[bands[2]]], axis = -1).astype(np.int16)
    
    rgb[rgb<0] = 0
    rgb[rgb > max_possible] = max_possible # Filter out saturation points at arbitrarily defined max_possible value
    
    rgb = rgb.astype(float)
    rgb *= 1 / np.max(rgb)
    ### > 
    
    ### < takes a T/F mask, apply a color to T areas  
    for mask, color in paint_on_mask:        
        rgb[mask] = np.array(color)/ 255.0
    ### > 
    
    
    fig, ax = plt.subplots(figsize = aspect_ratio_helper(*rgb.shape[:2], fixed_width = width))

    lat_formatter = FuncFormatter(lambda x, pos: round(dataset.latitude.values[pos] ,4) )
    lon_formatter = FuncFormatter(lambda x, pos: round(dataset.longitude.values[pos],4) )

    plt.ylabel("Latitude")
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.xlabel("Longitude")
    ax.xaxis.set_major_formatter(lon_formatter)
   
    if 'time' in dataset:
        plt.imshow((rgb[at_index]))
    else:
        plt.imshow(rgb)  
    
    plt.show()
