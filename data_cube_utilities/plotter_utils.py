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
    
def xarray_time_series_plot(dataset, plot_types, fig_params={'figsize':(12,6)}, component_plot_params={}, fit_params={}, fig=None):
    """
    Plot data variables in an xarray.Dataset together in one figure, 
    but with different plot types for each (e.g. box-and-whisker plot, line plot, scatter plot).
    
    Paramaeters
    -----------
    dataset: xarray.Dataset 
        A Dataset containing some bands like NDVI or WOFS.
        Must have coordinates: time, latitude, longitude.
    plot_types: dict
        Dictionary mapping names of DataArrays in the Dataset to plot to 
        their plot types (e.g. {'ndvi':'point', 'wofs':'line'}).
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}).
    component_plot_params: dict
        Dictionary mapping parameter names to dictionaries of matplotlib 
        formatting parameters for individual plots (e.g. {'ndvi':{'color':'red'}, 'wofs':{'color':'blue'}}).
    fit_params: dict
        Dictionary mapping parameter names to types fo curve fits. e.g. {'ndvi': 'gaussian'}.
        The curves fit to means along all dimesions except time to create a curve in the 2D plot.
        Curve types can be any of ['gaussian'].
    fig: matplotlib.figure.Figure
        The figure to use for the plot. The figure must have at least one Axes object.
        You can use the code ``fig,ax = plt.subplots()`` to create a figure with an associated Axes object.
        The code ``fig = plt.figure()`` will not provide the Axes object.
    """
    plotting_data = dataset.stack(lat_lon=('latitude', 'longitude'))
    if fig is None:
        fig, ax = plt.subplots(figsize=(9,6))
    else:
        ax = fig.axes[0]
    
    possible_time_agg_strs = ['week', 'month']
    time_agg_str = 'time'
    for possible_time_agg_str in possible_time_agg_strs:
        if possible_time_agg_str in list(plotting_data.coords):
            time_agg_str = possible_time_agg_str
            break
    times = plotting_data[time_agg_str].values
    times_no_nan = set()
    
    # Data variable plots.
    data_var_plots = {}
    for data_arr_name, plot_type in plot_types.items():
        if len(plotting_data[data_arr_name].values.shape) > 1:    
            formatted_data = xr.DataArray(np.full_like(plotting_data[data_arr_name].values, np.nan)) 
        else:
            formatted_data = xr.DataArray(np.full_like(plotting_data[data_arr_name].values.reshape(-1,1), np.nan)) 
        for i, time in enumerate(times):
            formatted_data.loc[i,:] = plotting_data.loc[{time_agg_str:time}][data_arr_name].values
        # Take a mean of all values for each time.
        plot_data = np.nanmean(formatted_data.values, axis=1)
        # Any times with all nan data are ignored in any plot type.
        nan_mask = ~np.isnan(plot_data)
        current_times = times[nan_mask]
        current_epochs = np.array(list(map(n64_to_epoch, current_times))) if time_agg_str == 'time' else None
        current_x_locs = current_epochs if time_agg_str == 'time' else current_times
        # Large scales for x_locs can break the curve fitting for some reason.
        current_x_locs = (current_x_locs - current_x_locs.min()) / (current_x_locs.max() - current_x_locs.min())
        times_no_nan.update(current_times)
        plot_params = component_plot_params.get(data_arr_name, {})
        # Create specified plot types.
        if plot_type == 'scatter':
            data_var_plots[data_arr_name] = ax.scatter(current_x_locs, plot_data[nan_mask], **plot_params)
        elif plot_type == 'box':
            boxplot_nan_mask = ~np.isnan(formatted_data)
            filtered_formatted_data = [] # Data formatted for matplotlib.pyplot.boxplot().
            for i, (d, m) in enumerate(zip(formatted_data, boxplot_nan_mask)):
                if len(d[m] != 0):
                    filtered_formatted_data.append(d.values[m])
            box_width = 0.5*np.min(np.diff(current_x_locs))
            boxprops = plot_params.pop('boxprops', dict(facecolor='orange'))
            flierprops = plot_params.pop('flierprops', dict(marker='o', markersize=0.25))
            bp = ax.boxplot(filtered_formatted_data, widths=[box_width]*len(filtered_formatted_data), 
                            positions=current_x_locs, patch_artist=True, boxprops=boxprops, flierprops=flierprops, 
                            manage_xticks=False, **plot_params) # `manage_xticks=False` to avoid excessive padding on the x-axis.
            data_var_plots[data_arr_name] = bp['boxes'][0]
    times_no_nan = sorted(list(times_no_nan))
    epochs = np.array(list(map(n64_to_epoch, times_no_nan))) if time_agg_str == 'time' else None
    x_locs = epochs if time_agg_str == 'time' else times_no_nan
    x_locs = (x_locs - x_locs.min()) / (x_locs.max() - x_locs.min())
    
    # Curve fitting.
    fit_plots = {}
    fit_labels = []
    for data_arr_name, fit_type in fit_params.items():
        if fit_type == 'gaussian':
            subset_dataset = dataset.sel(time=times_no_nan)[data_arr_name]
            non_time_dims = list(set(subset_dataset.dims)-{time_agg_str})
            means = subset_dataset.mean(dim=non_time_dims).values
            mean = np.nanmean(subset_dataset.values)
            sigma = np.nanstd(subset_dataset.values)
            def gaus(x,a,x0,sigma):
                return a*exp(-(x-x0)**2/(2*sigma**2))
            popt,pcov = curve_fit(gaus,x_locs,means,p0=[1,mean,sigma])
            x_smooth = np.linspace(x_locs.min(), x_locs.max(), 200)
            fit_plots[data_arr_name], = ax.plot(x_smooth, gaus(x_smooth,*popt), '-')
            fit_labels.append('Gaussian fit of {}'.format(data_arr_name))
    # Label the axes and create the legend.
    date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), times_no_nan))) if time_agg_str=='time' else\
                naive_months_ticks_by_week(times_no_nan) if time_agg_str=='week' else\
                month_ints_to_month_names(times_no_nan)
    plt.xticks(x_locs, date_strs, rotation=45, ha='right', rotation_mode='anchor')
    plt.legend(handles=[plot for plot in data_var_plots.values()]+[fit_plot for fit_plot in fit_plots.values()], 
               labels=list(plot_types.keys())+fit_labels, loc='best')
    plt.tight_layout()
    
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
    times = list(map(n64_to_epoch, times))
    times = np.array(times)
    times = np.sort(times)
    mean  = dataset.mean(dim=['latitude','longitude'],  skipna = True).values
    medians = dataset.median(dim=['latitude','longitude'], skipna = True)
    
    std_dev = np.nanstd(mean)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    #Shaded Area
    quarter = np.nanpercentile(
    dataset.values.reshape((
        landsat_dataset.dims['time'],
        landsat_dataset.dims['latitude'] * landsat_dataset.dims['longitude'])),
        25,
        axis = 1
    )
    three_quarters = np.nanpercentile(
    dataset.values.reshape((
        landsat_dataset.dims['time'],
        landsat_dataset.dims['latitude'] * landsat_dataset.dims['longitude'])),
        75,
        axis = 1
    )
    np.array(quarter)
    np.array(three_quarters)
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    fillcolor1='gray'
    fillcolor2='brown'
    fillalpha=0.4
    plt.fill_between(times, mean, quarter,  interpolate=False, color=fillcolor1, alpha=fillalpha,label="25th")
    plt.fill_between(times, mean, three_quarters, interpolate=False, color=fillcolor1, alpha=fillalpha,label="75th")
        
    #Medians
    plt.plot(times,medians,color="black",marker="o",linestyle='None', label = "Medians")
    
    #Linear Regression (on everything)
    #Data formatted in a way for needed for Guassian and Linear Regression
    #regression_list = full_linear_regression(dataset)
    #formatted_time, value = zip(*regression_list)
    #formatted_time = np.array(formatted_time)
    
    #The Actual Plot
    plt.plot(times,mean,color="blue",label="Mean")

    #Linear Regression (on mean)
    m, b = np.polyfit(times, mean, 1)
    plt.plot(times, m*times + b, '-', color="red",label="linear regression of mean",linewidth = 3.0)

    #Gaussian Curve
    b = gaussian(len(times), std_dev)
    ga = filters.convolve1d(mean, b/b.sum(),mode="reflect")
    x_smooth = np.linspace(times.min(),times.max(), 200)
    y_smooth = spline(times, ga, x_smooth)
    plt.plot(x_smooth, y_smooth, '-',label="Gaussian Smoothed of mean", alpha=1, color='limegreen',linewidth = 3.0)
    
    
    #Formatting
    ax.grid(color='k', alpha=0.1, linestyle='-', linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(tfmt))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
    plt.xticks(rotation=45, fontsize=fontsize)
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

## Misc ##

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