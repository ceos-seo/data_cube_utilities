import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import datacube as dc
import xarray as xr
import utils.data_cube_utilities.data_access_api as dc_api 
from utils.data_cube_utilities.dc_utilities import perform_timeseries_analysis
from utils.data_cube_utilities.dc_mosaic import ls7_unpack_qa
from rasterstats import zonal_stats
from scipy import stats, exp
from scipy.stats import norm
import pylab
import matplotlib as mpl
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.optimize import curve_fit
from sklearn import linear_model
from scipy.interpolate import spline
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import calendar, datetime, time
import pytz
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import warnings

from scipy.interpolate import interp1d

def impute_missing_data_1D(data1D):
    """
    This function returns the data in the same format as it was 
    passed in, but with missing values either masked out or imputed with appropriate values 
    (currently only using a linear trend). Many linear plotting functions for 1D data often 
    (and should) only connect contiguous,  non-nan data points. This leaves gaps in the 
    piecewise linear plot, which are sometimes graphically undesirable.
    
    Parameters
    ----------
    data: numpy.ndarray
        A 1D NumPy array for which missing values are to be masked or imputed 
        suitably for at least matplotlib plotting. If formatting for other libraries such 
        as seaborn or plotly is necessary, add that formatting requirement as a parameter.
    """
    nan_mask = ~np.isnan(data1D)
    x = np.arange(len(data1D))
    x_no_nan = x[nan_mask]
    data_no_nan = data1D[nan_mask]
    if len(x_no_nan) >= 2:
        f = interp1d(x_no_nan, data_no_nan)
        # Select points for interpolation.
        interpolation_x_mask = (x_no_nan[0]<=x) & (x<=x_no_nan[-1])
        interpolation_x = x[interpolation_x_mask]
        data1D_interp = np.arange(len(data1D), dtype=np.float32)
        # The ends of data1D may contain NaNs that must be included.
        end_nan_inds = x[(x<=x_no_nan[0]) | (x_no_nan[-1]<=x)]
        data1D_interp[end_nan_inds] = np.nan
        data1D_interp[interpolation_x_mask] = f(interpolation_x)
        return data1D_interp
    else: # Cannot interpolate with a single non-nan point.
        return data1D

## Datetime functions ##

def n64_to_epoch(timestamp):
    ts = pd.to_datetime(str(timestamp)) 
    time_format = "%Y-%m-%d"
    ts = ts.strftime(time_format)
    epoch = int(time.mktime(time.strptime(ts, time_format)))
    return epoch

def np_dt64_to_str(np_datetime, fmt='%Y-%m-%d'):
    """Converts a NumPy datetime64 object to a string based on a format string supplied to pandas strftime."""
    return pd.to_datetime(str(np_datetime)).strftime(fmt)

def tfmt(x, pos=None):
    return time.strftime("%Y-%m-%d",time.gmtime(x))

## End datetime functions ##

## Matplotlib colormap functions ##

def create_discrete_color_map(th, colors, alpha, cmap_name='my_cmap'):
    """
    Creates a discrete matplotlib LinearSegmentedColormap with thresholds for color changes.
    
    Parameters
    ----------
    th: list
        Threshold values. Must be between 0.0 and 1.0 - noninclusive.
    colors: list
        Colors to use between thresholds, so `len(colors) == len(th)+1`.
        Colors can be string names of matplotlib colors or 3-tuples of rgb values.
    alpha: float
        The alpha values to use for the colors, so `len(alpha) == len(colors)`.
    cmap_name: str
        The name of the colormap for matplotlib.
    """
    import matplotlib as mpl
    th = [0.0] + th + [1.0]
    cdict = {}
    # These are fully-saturated red, green, and blue - not the matplotlib colors for 'red', 'green', and 'blue'.
    primary_colors = ['red', 'green', 'blue'] 
    # Get the 3-tuples of rgb values for the colors.
    color_rgbs = [(mpl.colors.to_rgb(color) if isinstance(color,str) else color) for color in colors]
    # For each color entry to go into the color dictionary...
    for primary_color_ind, primary_color in enumerate(primary_colors):
        cdict_entry = [None]*len(th)
        # For each threshold (as well as 0.0 and 1.0), specify the values for this primary color.
        for row_ind, th_ind in enumerate(range(len(th))):
            # Get the two colors that this threshold corresponds to.
            th_color_inds = [0,0] if th_ind==0 else \
                            [len(colors)-1, len(colors)-1] if th_ind==len(th)-1 else \
                            [th_ind-1, th_ind]
            primary_color_vals = [color_rgbs[th_color_ind][primary_color_ind] for th_color_ind in th_color_inds]
            cdict_entry[row_ind] = (th[th_ind],) + tuple(primary_color_vals)
        cdict[primary_color] = cdict_entry
    cmap = LinearSegmentedColormap(cmap_name, cdict)
    return cmap

## End matplotlib colormap functions ##

def regression_massage(ds): 
    t_len = len(ds["time"])
    s_len = len(ds["latitude"]) * len(ds["longitude"])
    flat_values = ds.values.reshape(t_len * s_len)
    return list(zip(list(map(n64_to_epoch, ds.time.values)),flat_values))

def remove_nans(aList):
    i = 0
    while i < len(aList):
        if np.isnan(aList[i][1]):
            del aList[i]
            i = 0
        else:
            i+=1
    return aList

def full_linear_regression(ds):
    myList = regression_massage(ds)
    myList = remove_nans(myList)
    myList = sorted(myList, key=lambda tup: tup[0])
    time, value = zip(*myList)
    value = [int(x) for x in value]
    value = np.array(value)
    value.astype(int)
    time = np.array(time)
    time.astype(int)
    return list(zip(time,value))  
    
def xarray_plot_data_vars_over_time(dataset, colors=['orange', 'blue']):
    """
    Plot a line plot of all data variables in an xarray.Dataset on a shared set of axes.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        The Dataset containing data variables to plot. The only dimension and coordinate must be 'time'.
    colors: list
        A list of strings denoting colors for each data variable's points. 
        For example, 'red' or 'blue' are acceptable.
    """
    data_var_names = sorted(list(dataset.data_vars))
    len_dataset = dataset.time.size
    nan_mask = np.full(len_dataset, True)
    for i, data_arr_name in enumerate(data_var_names):
        data_arr = dataset[data_arr_name]
        nan_mask = nan_mask & data_arr.notnull().values
        plt.plot(data_arr[nan_mask], marker='o', c=colors[i])
    times = dataset.time.values
    date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), times)))
    plt.xticks(np.arange(len(date_strs[nan_mask])), date_strs[nan_mask], 
               rotation=45, ha='right', rotation_mode='anchor')
    plt.legend(data_var_names, loc='upper right')
    plt.show()
    
def xarray_scatterplot_data_vars(dataset, figure_kwargs={'figsize':(12,6)}, colors=['blue', 'orange'], markersize=5):
    """
    Plot a scatterplot of all data variables in an xarray.Dataset on a shared set of axes.
    Currently requires a 'time' coordinate, which constitutes the x-axis.

    Parameters
    ----------
    dataset: xarray.Dataset
        The Dataset containing data variables to plot.
    frac_dates: float
        The fraction of dates to label on the x-axis.
    figure_kwargs: dict
        A dictionary of kwargs for matplotlib figure creation.
    colors: list
        A list of strings denoting abbreviated colors for each data variable's points. 
        For example, 'r' is red and 'b' is blue.
    markersize: float
        The size of markers in the scatterplot.
    """
    plt.figure(**figure_kwargs)
    data_var_names = list(dataset.data_vars)
    len_dataset = dataset.time.size
    nan_mask = np.full(len_dataset, True)
    for i, data_arr in enumerate(dataset.data_vars.values()):
        if len(list(dataset.dims)) > 1:
            dims_to_check_for_nulls = [dim for dim in list(dataset.dims) if dim != 'time']
            nan_mask = nan_mask & data_arr.notnull().any(dim=dims_to_check_for_nulls).values 
        else:
            nan_mask = data_arr.notnull().values
        times = data_arr.to_dataframe().index.get_level_values('time').values
        plt.scatter(stats.rankdata(times, method='dense')-1, data_arr.values.flatten(), c=colors[i], s=markersize)
    unique_times = dataset.time.values
    date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), unique_times)))
    plt.xticks(np.arange(len(date_strs))[nan_mask], date_strs[nan_mask], 
               rotation=45, ha='right', rotation_mode='anchor')
    plt.xlabel('time')
    plt.legend(data_var_names, loc='upper right')
    plt.show()
    
def xarray_plot_ndvi_boxplot_wofs_lineplot_over_time(dataset, resolution=None, colors=['orange', 'blue']):
    """
    For an xarray.Dataset, plot a boxplot of NDVI and line plot of WOFS across time.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        A Dataset formatted as follows: 
            coordinates: time, latitude, longitude.
            data variables: ndvi, wofs
    resolution: str
        Denotes the resolution of aggregation. Only options are None or 'weekly'.
    colors: list
        A list of strings denoting colors for each data variable's points. 
        For example, 'red' or 'blue' are acceptable.
    """
    plotting_data = dataset.stack(lat_lon=('latitude', 'longitude'))
    time_agg_str = 'weekofyear' if resolution is not None and resolution == 'weekly' else 'time'
    if time_agg_str != 'time':
        plotting_data = plotting_data.groupby('time.'+time_agg_str).mean(dim='time')
    fig, ax = plt.subplots(figsize=(9,6))
    ndvi_box_color, wofs_line_color = ('orange', 'blue')
    times = plotting_data[time_agg_str].values
    
    # NDVI boxplot boxes
    # The data formatted for matplotlib.pyplot.boxplot().
    ndvi_formatted_data = xr.DataArray(np.full_like(plotting_data.ndvi.values, np.nan))
    for i, time in enumerate(times):
        ndvi_formatted_data.loc[i,:] = plotting_data.loc[{time_agg_str:time}].ndvi.values
    ndvi_nan_mask = ~np.isnan(ndvi_formatted_data)
    filtered_formatted_data = [] # Data formatted for matplotlib.pyplot.boxplot().
    acq_inds_to_keep = [] # Indices of acquisitions to keep. Other indicies contain all nan values.
    for i, (d, m) in enumerate(zip(ndvi_formatted_data, ndvi_nan_mask)):
        if len(d[m] != 0):
            filtered_formatted_data.append(d[m])
            acq_inds_to_keep.append(i)
    times_no_nan = times[acq_inds_to_keep]
    epochs = np.array(list(map(n64_to_epoch, times_no_nan))) if time_agg_str == 'time' else None
    x_locs = epochs if time_agg_str == 'time' else times_no_nan
    box_width = 0.5*np.min(np.diff(x_locs))
    bp = ax.boxplot(filtered_formatted_data, widths=[box_width]*len(filtered_formatted_data), 
                    positions=x_locs, patch_artist=True, boxprops=dict(facecolor=ndvi_box_color), 
                    flierprops=dict(marker='o', markersize=0.25), 
                    manage_xticks=False) # `manage_xticks=False` to avoid excessive padding on the x-axis.

    # WOFS line
    wofs_formatted_data = xr.DataArray(np.full_like(plotting_data.wofs.values, np.nan))
    for i, time in enumerate(times):
        wofs_formatted_data.loc[i,:] = plotting_data.loc[{time_agg_str:time}].wofs.values
    wofs_line_plot_data = np.nanmean(wofs_formatted_data.values, axis=1)
    wofs_nan_mask = ~np.isnan(wofs_line_plot_data)
    line = ax.plot(x_locs, wofs_line_plot_data[wofs_nan_mask], c=wofs_line_color)

    date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), times_no_nan))) if time_agg_str=='time' else \
                naive_months_ticks_by_week(times_no_nan)
    x_labels = date_strs
    plt.xticks(x_locs, x_labels, rotation=45, ha='right', rotation_mode='anchor')

    plt.legend(handles=[bp['boxes'][0],line[0]], labels=list(plotting_data.data_vars), loc='best')
    plt.tight_layout()
    plt.show()
    
def xarray_time_series_plot(dataset, plot_types, fig_params={'figsize':(18,12)}, component_plot_params={}, fit_params={}, 
                            scale_params={}, fig=None, ax=None, max_times_per_plot=None):
    """
    Plot data variables in an xarray.Dataset together in one figure, with different plot types for each 
    (e.g. box-and-whisker plot, line plot, scatter plot), and optional curve fitting to means or medians along time.
    Handles data binned with xarray.Dataset methods resample() and groupby(). That is, it handles data binned along time
    or across years (e.g. by week of year).
    
    Paramaeters
    -----------
    dataset: xarray.Dataset 
        A Dataset containing some bands like NDVI or WOFS.
        The primary coordinate must be 'time'.
    plot_types: dict
        Dictionary mapping names of DataArrays in the Dataset to plot to 
        their plot types (e.g. {'ndvi':'point', 'wofs':'line'}).
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}).
        Used to create a Figure ``if fig is None and ax is None``.
    component_plot_params: dict
        Dictionary mapping names of DataArrays to dictionaries of matplotlib 
        formatting parameters for individual plots (e.g. {'ndvi':{'color':'red'}, 'wofs':{'color':'blue'}}).
    fit_params: dict
        Dictionary mapping names of DataArrays to 2-tuples of strings in ['mean', 'median'] and types of curve fits. 
        e.g. {'ndvi': ('mean', 'gaussian')}. The curves fit to means or medians along all dimensions except time 
        to create a curve in the 2D plot. Curve types can be any of ['gaussian'].
        Note that in the case of multiple plots being created (see ``max_times_per_plot`` below), figsize will be the size
        of each plot - not the entire figure.
    scale_params: dict
        Dictionary mapping names of DataArrays to scaling methods (e.g. {'ndvi': None, 'wofs':'norm'}). 
        The options are [None, 'std', 'norm']. The option ``None``, which is the default, performs no scaling. 
        The option 'std' standardizes. The option 'norm' normalizes (min-max scales). 
        Note that of these options, only normalizing guarantees that the y values will be in a fixed range - namely [0,1].
    fig: matplotlib.figure.Figure
        The figure to use for the plot. The figure must have at least one Axes object.
        You can use the code ``fig,ax = plt.subplots()`` to create a figure with an associated Axes object.
        The code ``fig = plt.figure()`` will not provide the Axes object.
        The Axes object used will be the first. This is ignored if ``max_times_per_plot`` is less than the number of times.
    ax: matplotlib.axes.Axes
        The axes to use for the plot. This is ignored if ``max_times_per_plot`` is less than the number of times.
    max_times_per_plot: int
        The maximum number of times per plot. If specified, one plot will be generated for each group 
        of this many times. The plots will be arranged in a grid.
    """
    # Determine how the data was aggregated, if at all.
    possible_time_agg_strs = ['week', 'weekofyear', 'month']
    time_agg_str = 'time'
    for possible_time_agg_str in possible_time_agg_strs:
        if possible_time_agg_str in list(dataset.coords):
            time_agg_str = possible_time_agg_str
            break
    # Make the data 2D - time and a stack of all other dimensions.
    non_time_dims = list(set(dataset.dims)-{time_agg_str})
    all_plotting_data = dataset.stack(stacked_data=non_time_dims)
    all_times = all_plotting_data[time_agg_str].values
    # Mask out times for which no data variable to plot has any non-NaN data.
    nan_mask_data_vars = list(all_plotting_data[list(plot_types.keys())].notnull().data_vars.values())
    for i, data_var in enumerate(nan_mask_data_vars):
        time_nan_mask = data_var.values if i == 0 else time_nan_mask | data_var.values
    time_nan_mask = np.any(time_nan_mask, axis=1)
    times_not_all_nan = all_times[time_nan_mask]
    all_plotting_data = all_plotting_data.loc[{time_agg_str:times_not_all_nan}]
    
    # Handle the potential for multiple plots.
    max_times_per_plot = len(times_not_all_nan) if max_times_per_plot is None else max_times_per_plot
    num_plots = int(np.ceil(len(times_not_all_nan)/max_times_per_plot))
    subset_num_cols = 2
    subset_num_rows = int(np.ceil(num_plots / subset_num_cols))
    if num_plots > 1:
        figsize = fig_params.pop('figsize')
        fig = plt.figure(figsize=figsize, **fig_params)
    
    # Create each plot.
    for time_ind, fig_ind in zip(range(0, len(times_not_all_nan), max_times_per_plot), range(num_plots)):
        lower_time_bound_ind, upper_time_bound_ind = time_ind, min(time_ind+max_times_per_plot, len(times_not_all_nan))
        time_extents = times_not_all_nan[[lower_time_bound_ind, upper_time_bound_ind-1]]
        # Retrieve or create the axes if necessary.
        if len(times_not_all_nan) <= max_times_per_plot:
            ax = retrieve_or_create_ax(fig, ax, **fig_params)
        else:
            ax = fig.add_subplot(subset_num_rows, subset_num_cols, fig_ind + 1)
        fig_times_not_all_nan = times_not_all_nan[lower_time_bound_ind:upper_time_bound_ind]
        plotting_data = all_plotting_data.loc[{time_agg_str:fig_times_not_all_nan}]
        epochs = np.array(list(map(n64_to_epoch, fig_times_not_all_nan))) if time_agg_str == 'time' else None
        x_locs = np_min_max_scale(epochs if time_agg_str == 'time' else fig_times_not_all_nan)
        
        legend_labels = []
        
        # Data variable plots within each plot.
        data_arr_plots = {}
        for data_arr_name, plot_type in plot_types.items():
            full_data_arr_plotting_data = plotting_data[data_arr_name].values
            # Any times with all nan data are ignored in any plot type.
            data_arr_nan_mask = np.any(~np.isnan(full_data_arr_plotting_data), axis=1)
            # Add this data variable label to the legend if it has any data to plot.
            if np.any(data_arr_nan_mask):
                legend_labels.append(data_arr_name) 
            else:
                continue
            # Remove times with all nan data.
            data_arr_plotting_data = full_data_arr_plotting_data[data_arr_nan_mask]
    
            # Scale
            scaling = scale_params.get(data_arr_name, None)
            if scaling is not None:
                data_arr_plotting_data = np_scale(data_arr_plotting_data, full_data_arr_plotting_data, 
                                                  min_max=(0.0,1.0), scaling=scaling)

            data_arr_times = fig_times_not_all_nan[data_arr_nan_mask]
            data_arr_epochs = epochs[data_arr_nan_mask]
            # Large scales for x_locs can break the curve fitting for some reason.
            data_arr_x_locs = x_locs[data_arr_nan_mask]
            plot_params = component_plot_params.get(data_arr_name, {})
            # Create specified plot types.
            if plot_type == 'scatter':
                # Ignore warning about taking the mean of an empty slice.        
                means = ignore_warnings(np.nanmean, data_arr_plotting_data, axis=1)
                data_arr_plots[data_arr_name] = ax.scatter(data_arr_x_locs, means, **plot_params)
            elif plot_type == 'box':
                boxplot_nan_mask = ~np.isnan(data_arr_plotting_data)
                filtered_formatted_data = [] # Data formatted for matplotlib.pyplot.boxplot().
                for i, (d, m) in enumerate(zip(data_arr_plotting_data, boxplot_nan_mask)):
                    if len(d[m] != 0):
                        filtered_formatted_data.append(d[m])
                box_width = 0.5*np.min(np.diff(data_arr_x_locs)) if len(data_arr_x_locs) > 1 else 0.5
                # Provide default arguments.
                boxprops = plot_params.pop('boxprops', dict(facecolor='orange'))
                flierprops = plot_params.pop('flierprops', dict(marker='o', markersize=0.25))
                bp = ax.boxplot(filtered_formatted_data, widths=[box_width]*len(filtered_formatted_data), 
                                positions=data_arr_x_locs, patch_artist=True, boxprops=boxprops, flierprops=flierprops, 
                                manage_xticks=False, **plot_params) # `manage_xticks=False` to avoid excessive padding on the x-axis.
                data_arr_plots[data_arr_name] = bp['boxes'][0]

        # Curve fitting.
        fit_plots = {}
        fit_labels = []
        for data_arr_name, (agg_type, fit_type) in fit_params.items():
            # First consider only times for which at least one DataArray did not have all NaN values.
            # Note that the dimensions of variables here are likely smaller than in the primary plotting loop above.
            full_data_arr_plotting_data = plotting_data[data_arr_name].values
            # This particular DataArray may still have all-NaN slices, so we mask before plotting.
            data_arr_nan_mask = np.any(~np.isnan(full_data_arr_plotting_data), axis=1)
            data_arr_plotting_data = full_data_arr_plotting_data[data_arr_nan_mask]
            data_arr_x_locs = x_locs[data_arr_nan_mask]
            if agg_type == 'mean':
                y = ignore_warnings(np.nanmean, data_arr_plotting_data, axis=1)
            elif agg_type == 'median':
                y = ignore_warnings(np.nanmedian, data_arr_plotting_data, axis=1)
            # Handle differences in plotting methods.
            if fit_type == 'gaussian':
                if len(data_arr_x_locs) < 3:
                    continue # 3 data points are needed to determine a unique Gaussian.
                fit_plots[data_arr_name] = plot_gaussian(data_arr_x_locs, y, ax=ax)
                fit_labels.append('Gaussian fit of {} of {}'.format(agg_type, data_arr_name))
        # Label the axes and create the legend.
        date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), fig_times_not_all_nan))) if time_agg_str=='time' else\
                    naive_months_ticks_by_week(fig_times_not_all_nan) if time_agg_str in ['week', 'weekofyear'] else\
                    month_ints_to_month_names(fig_times_not_all_nan)
        plt.xticks(x_locs, date_strs, rotation=45, ha='right', rotation_mode='anchor')
        plt.legend(handles=[plot for plot in data_arr_plots.values()]+[fit_plot for fit_plot in fit_plots.values()], 
                   labels=legend_labels+fit_labels, loc='best')
        plt.title("Figure {}: Time Range {} to {}".format(fig_ind, date_strs[0], date_strs[-1]))
        plt.tight_layout()

def retrieve_or_create_ax(fig=None, ax=None, **fig_params):
    """
    Returns an appropriate Axes object given possible Figure or Axes objects.
    If neither is supplied, a new figure will be created with associated axes.
    """
    if fig is None:
        if ax is None:
            fig, ax = plt.subplots(**fig_params)
    else:
        ax = fig.axes[0]
    return ax
    
## Begin curve fitting ##

def gauss(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))
    
def plot_gaussian(x, y, n_pts=200, fig_params={'figsize':(12,6)}, plotting_kwargs={'linestyle': '-'}, fig=None, ax=None):
    """
    Parameters
    ----------
    x: np.ndarray
        A 1D NumPy array. The x values to fit to.
    y: np.ndarray
        A 1D NumPy array. The y values to fit to.
    n_pts: int
        The number of points to use for the smoothed fit. More will result in a smoother curve.
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}).
        Used to create a Figure ``if fig is None and ax is None``.
    plotting_kwargs: dict
        The kwargs for the call to ``matplotlib.axes.Axes.plot()``.
    fig: matplotlib.figure.Figure
        The figure to use for the plot. The figure must have at least one Axes object.
        You can use the code ``fig,ax = plt.subplots()`` to create a figure with an associated Axes object.
        The code ``fig = plt.figure()`` will not provide the Axes object. 
        The Axes object used will be the first.
    ax: matplotlib.axes.Axes
        The axes to use for the plot.
        
    Returns
    -------
    lines: matplotlib.lines.Line2D
        Can be used as a handle for a matplotlib legend (i.e. plt.legend(handles=...)) among other things.
    """
    # Retrieve or create the axes if necessary.
    ax = retrieve_or_create_ax(fig, ax, **fig_params)
    
    mean, sigma = np.nanmean(y), np.nanstd(y)
    popt,pcov = curve_fit(gauss,x,y,p0=[1,mean,sigma], maxfev=np.iinfo(np.int32).max)
    x_smooth = np.linspace(x.min(), x.max(), n_pts)
    return ax.plot(x_smooth, gauss(x_smooth,*popt), **plotting_kwargs)[0]
    
## End curve fitting ##
    
def plot_band(landsat_dataset, dataset, figsize=(20,15), fontsize=24, legend_fontsize=24):
    """
    Plots several statistics over time - including mean, median, linear regression of the 
    means, Gaussian smoothed curve of means, and the band enclosing the 25th percentiles 
    and the 75th percentiles. This is very similar to the output of the Comet Time Series 
    Toolset (https://github.com/CosmiQ/CometTS). 
    
    Parameters
    ----------
    landsat_dataset: xarray.Dataset
        An xarray `Dataset` containing longitude, latitude, and time coordinates.
    dataset: xarray.DataArray
        An xarray `DataArray` containing time, latitude, and longitude coordinates.
    figsize: tuple
        A 2-tuple of the figure size in inches for the entire figure.
    fontsize: int
        The font size to use for text.
    """
    
    #Calculations
    times = dataset.time.values
    epochs = np.sort(np.array(list(map(n64_to_epoch, times))))
    x_locs = (epochs - epochs.min()) / (epochs.max() - epochs.min())
    means  = dataset.mean(dim=['latitude','longitude'],  skipna = True).values
    medians = dataset.median(dim=['latitude','longitude'], skipna = True).values
    mask = ~np.isnan(means) & ~np.isnan(medians)
    
    plt.figure(figsize=figsize)
    ax = plt.gca()

    #Shaded Area
    with warnings.catch_warnings():
        # Ignore warning about encountering an All-NaN slice. Some acquisitions have all-NaN values.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        quarter = np.nanpercentile(
        dataset.values.reshape((
            len(dataset['time']),
            len(dataset['latitude']) * len(dataset['longitude']))),
            25,
            axis = 1
        )
        three_quarters = np.nanpercentile(
        dataset.values.reshape((
            len(dataset['time']),
            len(dataset['latitude']) * len(dataset['longitude']))),
            75,
            axis = 1
        )
    np.array(quarter)
    np.array(three_quarters)
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    fillcolor='gray'
    fillalpha=0.4
    plt.fill_between(x_locs, means, quarter,  interpolate=False, color=fillcolor, alpha=fillalpha,label="25th")
    plt.fill_between(x_locs, means, three_quarters, interpolate=False, color=fillcolor, alpha=fillalpha,label="75th")
        
    #Medians
    plt.plot(x_locs,medians,color="black",marker="o",linestyle='None', label = "Medians")
    
    #Linear Regression (on everything)
    #Data formatted in a way for needed for Guassian and Linear Regression
    
    #The Actual Plot
    
    plt.plot(x_locs,means,color="blue",label="Mean")

    #Linear Regression (on mean)
    m, b = np.polyfit(x_locs[mask], means[mask], 1)
    plt.plot(x_locs, m*x_locs + b, '-', color="red",label="linear regression of measn",linewidth = 3.0)

    #Gaussian Curve
    plot_gaussian(x_locs[mask], means[mask], ax=ax,
                  plotting_kwargs=dict(linestyle='-', label="Gaussian Smoothed of means", 
                                       alpha=1, color='limegreen',linewidth = 3.0))
    
    #Formatting
    date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), times[mask])))
    ax.grid(color='k', alpha=0.1, linestyle='-', linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(tfmt))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
    plt.xticks(x_locs, date_strs, rotation=45, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_ylabel('Value', fontsize=fontsize)
    plt.show()
    
def plot_pixel_qa_value(dataset, platform, values_to_plot, bands = "pixel_qa", plot_max = False, plot_min = False):
    times = dataset.time.values
    mpl.style.use('seaborn')
    plt.figure(figsize=(20,15))
    quarters = []
    three_quarters = []
    percentiles = []
   
    for i,v in enumerate(values_to_plot):
        _xarray  = ls7_unpack_qa(dataset.pixel_qa, values_to_plot[i])
        y = _xarray.mean(dim= ['latitude', 'longitude'])
        times = dataset.time.values.astype(float)
        std_dev = np.std(y)
        std_dev = std_dev.values
        b = gaussian(len(times), std_dev)
        ga = filters.convolve1d(y, b/b.sum(),mode="reflect")
        ga=interpolate_gaps(ga, limit=3)
        plt.plot(times, ga, '-',label="Gaussian ", alpha=1, color='black')
        
        x_smooth = np.linspace(times.min(),times.max(), 200)
        y_smooth = spline(times, ga, x_smooth)
        plt.plot(x_smooth, y_smooth, '-',label="Gaussian Smoothed", alpha=1, color='cyan')
        
        for i, q in enumerate(_xarray):
            quarters.append(np.nanpercentile(_xarray, 25))
            three_quarters.append(np.nanpercentile(_xarray, 75))
            #print(q.values.mean())
        
        ax = plt.gca()
        ax.grid(color='lightgray', linestyle='-', linewidth=1)
        fillcolor='gray'
        fillalpha=0.4
        linecolor='gray'
        linealpha=0.6
        plt.fill_between(times, y, quarters,  interpolate=False, color=fillcolor, alpha=fillalpha)
        plt.fill_between(times, y, three_quarters, interpolate=False, color=fillcolor, alpha=fillalpha)
        plt.plot(times,quarters,color=linecolor , alpha=linealpha)
        plt.plot(times,three_quarters,color=linecolor, alpha=linealpha)
        
        medians = _xarray.median(dim=['latitude','longitude'])
        plt.scatter(times,medians,color='mediumpurple', label="medians", marker="D")
        
        m, b = np.polyfit(times, y, 1)
        plt.plot(times, m*times + b, '-', color="red",label="linear regression")
        plt.style.use('seaborn')
        
        plt.plot(times, y, marker="o")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(rotation=90)    



## Misc ##

def xarray_values_in(data, values, data_vars=None):
    """
    Returns a mask for an xarray Dataset or DataArray, with True wherever the value is in values.
    
    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        The data to check for value matches.
    values: list-like
        The values to check for.
    data_vars: list-like
        The data variables to check - given by name.
    
    Returns
    -------
    mask: np.ndarray
        A NumPy array shaped like ``data``. The mask can be used to mask ``data``.
        That is, ``data.where(mask)`` is an intended use.
    """
    if isinstance(data, xr.Dataset):
        mask = np.full_like(list(data.data_vars.values())[0], False, dtype=np.bool)
        data_arr_names = list(data.data_vars) if data_vars is None else data_vars
        for data_arr_name in data_arr_names:
            data_arr = data[data_arr_name]
            for value in values:
                mask = mask | (data_arr.values == value)
    elif isinstance(data, xr.DataArray):
        mask = np.full_like(data, False, dtype=np.bool)
        for value in values:
            mask = mask | (data.values == value)
    return mask

def ignore_warnings(func, *args, **kwargs):
    """Runs a function while ignoring warnings"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = func(*args, **kwargs)
    return ret

def np_scale(arr, pop_arr=None, pop_min_max=None, mean_std=None, min_max=None, scaling='norm'):
    """
    Scales a NumPy array with standard scaling or norm scaling.
    
    Parameters
    ----------
    arr: numpy.ndarray
        The NumPy array to scale.
    pop_arr: numpy.ndarray
        The NumPy array to treat as the population. 
        If specified, all members of arr must be within pop_arr or min_max must be specified.
    pop_min_max: tuple
        A 2-tuple of the population minimum and maximum, in that order. 
        Supercedes pop_arr when normalizing.
    mean_std: tuple
        A 2-tuple of the population mean and standard deviation, in that order. 
        Supercedes pop_arr when standard scaling.
    min_max: tuple
        A 2-tuple which specifies the desired range of the final output - the minimum and the maximum, in that order.
        If all values are the same after standardizing or normalizing, all values will become min_max[0].
    scaling: str
        The options are ['std', 'norm']. 
        The option 'std' standardizes. The option 'norm' normalizes (min-max scales). 
    
    Raises
    ------
    ``ValueError`` 
        If the maximum and minimum values are identical or the maximum is smaller than the minimum.
    """
    pop_arr = arr if pop_arr is None else pop_arr
    if scaling == 'norm':
        pop_min, pop_max = (pop_min_max[0], pop_min_max[1]) if pop_min_max is not None else (pop_arr.min(), pop_arr.max())
        numerator, denominator = arr - pop_min, pop_max - pop_min
    elif scaling == 'std':
        mean, std = mean_std if mean_std is not None else pop_arr.mean(), pop_arr.std()
        numerator, denominator = arr - mean, std
    # Primary scaling
    new_arr = arr
    scaling_performed = False
    if denominator - 0.0 > 0.00001:
        new_arr = numerator / denominator
        scaling_performed = True
    # Optional final scaling.
    if min_max is not None:
        new_arr = np.interp(new_arr, (new_arr.min(), new_arr.max()), min_max) if scaling_performed else \
                  np.full_like(new_arr, min_max[0])
    return new_arr

def remove_non_unique_ordered_list_str(ordered_list):
    """
    Sets all occurrences of a value in an ordered list after its first occurence to ''.
    For example, ['a', 'a', 'b', 'b', 'c'] would become ['a', '', 'b', '', 'c'].
    """
    prev_unique_str = ""
    for i in range(len(ordered_list)):
        current_str = ordered_list[i]
        if current_str != prev_unique_str:
            prev_unique_str = current_str
        else:
            ordered_list[i] = ""
    return ordered_list

# For February, assume leap years are included.
days_per_month = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 
                  7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def get_weeks_per_month(num_weeks):
    """
    Including January, give 5 weeks to every third month - accounting for 
    variation between 52 and 54 weeks in a year by adding weeks to the last 3 months.
    """
    last_months_num_weeks = None
    if num_weeks <= 52:
        last_months_num_weeks = [5,4,4]
    elif num_weeks == 53:
        last_months_num_weeks = [5,4,5]
    elif num_weeks == 54:
        last_months_num_weeks = [5,5,5]
    return {month_int:num_weeks for (month_int,num_weeks) in zip(days_per_month.keys(), [5,4,4]*3+last_months_num_weeks)}

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def month_ints_to_month_names(month_ints):
    """
    Converts ordinal numbers for months (in range [1,12]) to their 3-letter names.
    """
    return [month_names[i-1] for i in month_ints]

def week_ints_to_month_names(week_ints):
    """
    Converts ordinal numbers for weeks (in range [1,54]) to their months' 3-letter names.
    """
    weeks_per_month = get_weeks_per_month(max(week_ints))
    week_month_strs = []
    for week_int in week_ints:
        month_int = -1
        for current_month_int, current_month_weeks in weeks_per_month.items():
            week_int -= current_month_weeks
            if week_int <= 0:
                month_int = current_month_int
                break
        week_month_strs.append(month_names[month_int-1])
    return week_month_strs

def naive_months_ticks_by_week(week_ints=None):
    """
    Given a list of week numbers (in range [1,54]), returns a list of month strings separated by spaces.
    Covers 54 weeks if no list-like of week numbers is given.
    This is only intended to be used for labeling axes in plotting.
    """
    month_ticks_by_week = []
    if week_ints is None: # Give month ticks for all weeks.
        month_ticks_by_week = week_ints_to_month_names(list(range(54)))
    else:
        month_ticks_by_week = remove_non_unique_ordered_list_str(week_ints_to_month_names(week_ints))
    return month_ticks_by_week