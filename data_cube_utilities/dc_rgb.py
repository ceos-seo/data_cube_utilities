import matplotlib.pyplot as plt
from time import time
import numpy as np

from .plotter_utils import figure_ratio, xarray_set_axes_labels, retrieve_or_create_fig_ax

# Change the bands (RGB) here if you want other false color combinations
def rgb(dataset, at_index = 0, bands = ['red', 'green', 'blue'], paint_on_mask = [],
        min_possible = 0, max_possible = 10000, width = 10, fig=None, ax=None):
    """
    Creates a figure showing an area, using three specified bands as the rgb componenets.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        A Dataset containing at least latitude and longitude coordinates and optionally time.
        The coordinate order should be time, latitude, and finally longitude.
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
    fig: matplotlib.figure.Figure
        The figure to use for the plot.
        If only `fig` is supplied, the Axes object used will be the first.
    ax: matplotlib.axes.Axes
        The axes to use for the plot.
        
    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes used for the plot.
    """
    ### < Dataset to RGB Format, needs float values between 0-1 
    rgb = np.stack([dataset[bands[0]],
                    dataset[bands[1]],
                    dataset[bands[2]]], axis = -1)
    # Interpolate values to be in the range [0,1] for creating the image.
    rgb = np.interp(rgb, (np.nanmin(rgb), np.nanmax(rgb)), [0,1])
    ### > 
    
    ### < takes a T/F mask, apply a color to T areas  
    for mask, color in paint_on_mask:        
        rgb[mask] = np.array(color)/ 255.0
    ### > 
    
    fig, ax = retrieve_or_create_fig_ax(fig, ax, figsize=figure_ratio(rgb.shape[:2], fixed_width = width))

    xarray_set_axes_labels(dataset, ax)
   
    if 'time' in dataset.dims:
        ax.imshow(rgb[at_index])
    else:
        ax.imshow(rgb)
    
    return fig, ax