import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
import matplotlib as mpl
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.interpolate import CubicSpline
from scipy.interpolate import spline
from matplotlib.ticker import FuncFormatter
import  time
from matplotlib.colors import LinearSegmentedColormap
from copy import copy
from scipy import stats
import warnings

from .dc_mosaic import ls7_unpack_qa
from .curve_fitting import gaussian_fit, poly_fit
from .scale import xr_scale, np_scale
from .dc_utilities import ignore_warnings

from .curve_fitting import gaussian_fit, poly_fit
from .scale import xr_scale, np_scale
from .dc_utilities import ignore_warnings

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

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
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

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
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

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
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

def xarray_time_series_plot(dataset, plot_descs, fig_params={'figsize':(18,12)}, scale_params={}, fig=None, ax=None, max_times_per_plot=None, show_legend=True):
    """
    Plot data variables in an xarray.Dataset together in one figure, with different plot types for each
    (e.g. box-and-whisker plot, line plot, scatter plot), and optional curve fitting to means or medians along time.
    Handles data binned with xarray.Dataset methods resample() and groupby(). That is, it handles data binned along time
    or across years (e.g. by week of year).

    Parameters
    -----------
    dataset: xarray.Dataset
        A Dataset containing some bands like NDVI or WOFS.
        The primary coordinate must be 'time'.
    plot_descs: dict
        Dictionary mapping names of DataArrays in the Dataset to plot to dictionaries mapping aggregation types (e.g. 'mean', 'median')
        to lists of dictionaries mapping plot types (e.g. 'line', 'box', 'scatter') to keyword arguments for plotting.

        Aggregation happens within time slices and can be many-to-many or many-to-one. Some plot types require many-to-many aggregation, and some other plot types require many-to-one aggregation.
        Aggregation types can be any of ['mean', 'median', 'none'], with 'none' performing no aggregation.

        Plot types can be any of ['scatter', 'line', 'gaussian', 'poly', 'cubic_spline', 'box'].
        The plot type 'poly' requires a 'degree' entry mapping to an integer in its dictionary of keyword arguments.

        Here is an example:
        {'ndvi':       {'mean': [{'line': {'color': 'forestgreen', 'alpha':alpha}}],
                        'none':  [{'box': {'boxprops': {'facecolor':'forestgreen', 'alpha':alpha},
                                                        'showfliers':False}}]}}
        This example will create a green line plot of the mean of the 'ndvi' band as well as a green box plot of the 'ndvi' band.
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}). Used to create a Figure ``if fig is None and ax is None``.
        Note that in the case of multiple plots being created (see ``max_times_per_plot`` below), figsize will be the size
        of each plot - not the entire figure.
    scale_params: dict
        Currently not used.
        Dictionary mapping names of DataArrays to scaling methods (e.g. {'ndvi': 'std', 'wofs':'norm'}).
        The options are ['std', 'norm']. The option 'std' standardizes. The option 'norm' normalizes (min-max scales).
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
    show_legend: bool
        Whether or not to show the legend.

    Raises
    ------
    ValueError:
        If an aggregation type is not possible for a plot type

    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    # Lists of plot types that can and cannot accept many-to-one aggregation for each time slice.
    plot_types_requiring_aggregation = ['line', 'gaussian', 'poly', 'cubic_spline']
    plot_types_handling_aggregation = ['scatter'] + plot_types_requiring_aggregation
    plot_types_not_handling_aggregation = ['box']
    all_plot_types = plot_types_requiring_aggregation + plot_types_handling_aggregation + plot_types_not_handling_aggregation

    # Aggregation types that aggregate all values for a given time to one value.
    many_to_one_agg_types = ['mean', 'median']
    # Aggregation types that aggregate to many values or do not aggregate.
    many_to_many_agg_types = ['none']
    all_agg_types = many_to_one_agg_types + many_to_many_agg_types


    # Determine how the data was aggregated, if at all.
    possible_time_agg_strs = ['week', 'weekofyear', 'month']
    time_agg_str = 'time'
    for possible_time_agg_str in possible_time_agg_strs:
        if possible_time_agg_str in list(dataset.coords):
            time_agg_str = possible_time_agg_str
            break
    # Make the data 2D - time and a stack of all other dimensions.
    non_time_dims = list(set(dataset.dims)-{time_agg_str})
    all_plotting_bands = list(plot_descs.keys())
    all_plotting_data = dataset[all_plotting_bands].stack(stacked_data=non_time_dims)
    all_times = all_plotting_data[time_agg_str].values
    # Mask out times for which no data variable to plot has any non-NaN data.
    nan_mask_data_vars = list(all_plotting_data[all_plotting_bands].notnull().data_vars.values())
    for i, data_var in enumerate(nan_mask_data_vars):
        time_nan_mask = data_var.values if i == 0 else time_nan_mask | data_var.values
    time_nan_mask = np.any(time_nan_mask, axis=1)
    times_not_all_nan = all_times[time_nan_mask]
    all_plotting_data = all_plotting_data.loc[{time_agg_str:times_not_all_nan}]

    # Scale
    if isinstance(scale_params, str): # if scale_params denotes the scaling type for the whole Dataset, scale the Dataset.
        all_plotting_data = xr_scale(all_plotting_data, scaling=scale_params)
    elif len(scale_params) > 0: # else, it is a dictionary denoting how to scale each DataArray.
        for data_arr_name, scaling in scale_params.items():
            all_plotting_data[data_arr_name] = xr_scale(all_plotting_data[data_arr_name], scaling=scaling)

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
        x_locs = np_scale(epochs if time_agg_str == 'time' else fig_times_not_all_nan)

        # Data variable plots within each plot.
        data_arr_plots = []
        legend_labels = []
        # For each data array to plot...
        for data_arr_name, agg_dict in plot_descs.items():
            # For each aggregation type (e.g. 'mean', 'median')...
            for agg_type, plot_dicts in agg_dict.items():
                # For each plot for this aggregation type...
                for plot_dict in plot_dicts:
                    for plot_type, plot_kwargs in plot_dict.items():
                        assert plot_type in all_plot_types, \
                            r"For the '{}' DataArray: plot_type '{}' not recognized".format(data_arr_name, plot_type)
                        full_data_arr_plotting_data = plotting_data[data_arr_name].values
                        # Any times with all nan data are ignored in any plot type.
                        data_arr_nan_mask = np.any(~np.isnan(full_data_arr_plotting_data), axis=1)

                        # Skip plotting this data variable if it does not have enough data to plot.
                        if skip_plot(np.sum(data_arr_nan_mask), plot_type, plot_kwargs):
                            continue

                        # Remove times with all nan data.
                        data_arr_plotting_data = full_data_arr_plotting_data[data_arr_nan_mask]
                        # Large scales for x_locs can break the curve fitting for some reason.
                        data_arr_x_locs = x_locs[data_arr_nan_mask]

                        # Some plot types require aggregation.
                        if plot_type in plot_types_requiring_aggregation:
                            if agg_type not in many_to_one_agg_types:
                                raise ValueError("For the '{}' DataArray: the plot type '{}' requires aggregation "
                                                 "(currently using '{}'). Please pass any of {} as the aggregation type "
                                                 "or change the plot type.".format(data_arr_name, plot_type, agg_type, many_to_one_agg_types))
                        # Some plot types cannot accept many-to-one aggregation.
                        if plot_type not in plot_types_handling_aggregation:
                            if agg_type not in many_to_many_agg_types:
                                raise ValueError("For the '{}' DataArray: the plot type '{}' doesn't accept aggregation "
                                                 "(currently using '{}'). Please pass any of {} as the aggregation type "
                                                 "or change the plot type.".format(data_arr_name, plot_type, agg_type, many_to_many_agg_types))

                        if agg_type == 'mean':
                            y = ignore_warnings(np.nanmean, data_arr_plotting_data, axis=1)
                        elif agg_type == 'median':
                            y = ignore_warnings(np.nanmedian, data_arr_plotting_data, axis=1)
                        elif agg_type == 'none':
                            y = data_arr_plotting_data

                        # Create specified plot types.
                        plot_type_str = "" # Used to label the legend.
                        if plot_type == 'scatter':
                            # Ignore warning about taking the mean of an empty slice.
                            data_arr_plots.append(ax.scatter(data_arr_x_locs, y, **plot_kwargs))
                            plot_type_str += 'scatterplot'
                        elif plot_type == 'line':
                            data_arr_plots.append(ax.plot(data_arr_x_locs, y, **plot_kwargs)[0])
                            plot_type_str += 'lineplot'
                        elif plot_type == 'box':
                            boxplot_nan_mask = ~np.isnan(y)
                            filtered_formatted_data = [] # Data formatted for matplotlib.pyplot.boxplot().
                            for i, (d, m) in enumerate(zip(y, boxplot_nan_mask)):
                                if len(d[m] != 0):
                                    filtered_formatted_data.append(d[m])
                            box_width = 0.5*np.min(np.diff(data_arr_x_locs)) if len(data_arr_x_locs) > 1 else 0.5
                            # Provide default arguments.
                            plot_kwargs.setdefault('boxprops', dict(facecolor='orange'))
                            plot_kwargs.setdefault('flierprops', dict(marker='o', markersize=0.5))
                            plot_kwargs.setdefault('showfliers', False)
                            bp = ax.boxplot(filtered_formatted_data, widths=[box_width]*len(filtered_formatted_data),
                                            positions=data_arr_x_locs, patch_artist=True,
                                            manage_xticks=False, **plot_kwargs) # `manage_xticks=False` to avoid excessive padding on the x-axis.
                            data_arr_plots.append(bp['boxes'][0])
                            plot_type_str += 'boxplot'
                        elif plot_type == 'gaussian':
                            data_arr_plots.append(plot_curvefit(data_arr_x_locs, y, fit_type=plot_type, plot_kwargs=plot_kwargs, ax=ax))
                            plot_type_str += 'gaussian fit'
                        elif plot_type == 'poly':
                            assert 'degree' in plot_kwargs, r"For the '{}' DataArray: When using 'poly' as the fit type," \
                                                            "the fit kwargs must have 'degree' specified.".format(data_arr_name)
                            data_arr_plots.append(plot_curvefit(data_arr_x_locs, y, fit_type=plot_type, plot_kwargs=plot_kwargs, ax=ax))
                            plot_type_str += 'degree {} polynomial fit'.format(plot_kwargs['degree'])
                        elif plot_type == 'cubic_spline':
                            data_arr_plots.append(plot_curvefit(data_arr_x_locs, y, fit_type=plot_type, plot_kwargs=plot_kwargs, ax=ax))
                            plot_type_str += 'cubic spline fit'
                        plot_type_str += ' of {}'.format(agg_type) if agg_type != 'none' else ''
                        legend_labels.append('{} of {}'.format(plot_type_str, data_arr_name))


        # Label the axes and create the legend.
        date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), fig_times_not_all_nan))) if time_agg_str=='time' else\
                    naive_months_ticks_by_week(fig_times_not_all_nan) if time_agg_str in ['week', 'weekofyear'] else\
                    month_ints_to_month_names(fig_times_not_all_nan)
        plt.xticks(x_locs, date_strs, rotation=45, ha='right', rotation_mode='anchor')
        if show_legend:
            plt.legend(handles=data_arr_plots, labels=legend_labels, loc='best')
        plt.title("Figure {}: Time Range {} to {}".format(fig_ind, date_strs[0], date_strs[-1]))
        plt.tight_layout()

## Curve fitting ##

def plot_curvefit(x, y, fit_type, x_smooth=None, n_pts=200, fig_params={}, plot_kwargs={}, fig=None, ax=None):
    """
    Plots a curve fit given x values, y values, a type of curve to plot, and parameters for that curve.

    Parameters
    ----------
    x: np.ndarray
        A 1D NumPy array. The x values to fit to.
    y: np.ndarray
        A 1D NumPy array. The y values to fit to.
    fit_type: str
        The type of curve to fit. One of ['poly', 'gaussian', 'cubic_spline'].
        The option 'poly' plots a polynomial fit. The option 'gaussian' plots a Gaussian fit.
        The option 'cubic_spline' plots a cubic spline fit.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}).
        Used to create a Figure ``if fig is None and ax is None``.
    plot_kwargs: dict
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
    # Avoid modifying the original arguments.
    fig_params = copy(fig_params)
    plot_kwargs = copy(plot_kwargs)

    fig_params.setdefault('figsize', (12,6))
    plot_kwargs.setdefault('linestyle', '-')

    # Retrieve or create the axes if necessary.
    ax = retrieve_or_create_ax(fig, ax, **fig_params)
    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), n_pts)
    if fit_type == 'gaussian':
        y_smooth = gaussian_fit(x, y, x_smooth)
    elif fit_type == 'poly':
        assert 'degree' in plot_kwargs.keys(), "When plotting a polynomal fit, there must be" \
                                              "a 'degree' entry in the plot_kwargs parameter."
        degree = plot_kwargs.pop('degree')
        y_smooth = poly_fit(x, y, degree, x_smooth)
    elif fit_type == 'cubic_spline':
        cs = CubicSpline(x,y)
        y_smooth = cs(x_smooth)
    return ax.plot(x_smooth, y_smooth, **plot_kwargs)[0]

## End curve fitting ##

def plot_band(dataset, figsize=(20,15), fontsize=24, legend_fontsize=24):
    """
    Plots several statistics over time - including mean, median, linear regression of the 
    means, Gaussian smoothed curve of means, and the band enclosing the 25th and 75th percentiles.
    This is very similar to the output of the Comet Time Series Toolset (https://github.com/CosmiQ/CometTS).
    
    Parameters
    ----------
    dataset: xarray.DataArray
        An xarray `DataArray` containing time, latitude, and longitude coordinates.
    figsize: tuple
        A 2-tuple of the figure size in inches for the entire figure.
    fontsize: int
        The font size to use for text.
    """
    # Calculations
    times = dataset.time.values
    epochs = np.sort(np.array(list(map(n64_to_epoch, times))))
    x_locs = (epochs - epochs.min()) / (epochs.max() - epochs.min())
    means  = dataset.mean(dim=['latitude','longitude'],  skipna = True).values
    medians = dataset.median(dim=['latitude','longitude'], skipna = True).values
    mask = ~np.isnan(means) & ~np.isnan(medians)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Shaded Area (percentiles)
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
    plt.fill_between(x_locs, quarter, three_quarters,  interpolate=False, color=fillcolor, alpha=fillalpha,
                     label="25th and 75th percentile band")
        
    #Medians
    plt.plot(x_locs,medians,color="black",marker="o",linestyle='None', label = "Medians")
    
    #The Actual Plot
    plt.plot(x_locs,means,color="blue",label="Mean")

    #Linear Regression (on mean)
    m, b = np.polyfit(x_locs[mask], means[mask], 1)
    plt.plot(x_locs, m*x_locs + b, '-', color="red",label="linear regression of means",linewidth = 3.0)

    #Gaussian Curve
    plot_curvefit(x_locs[mask], means[mask], fit_type='gaussian', ax=ax,
                  plot_kwargs=dict(linestyle='-', label="Gaussian smoothed of means",
                                   alpha=1, color='limegreen', linewidth = 3.0))
    
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

## Matplotlib colormap functions ##

def create_discrete_color_map(th, colors, data_range=[0,1], cmap_name='my_cmap'):
    """
    Creates a discrete matplotlib LinearSegmentedColormap with thresholds for color changes.

    Parameters
    ----------
    th: list
        Threshold values. Must be in the range of `data_range` - noninclusive.
    colors: list
        Colors to use between thresholds, so `len(colors) == len(th)+1`.
        Colors can be string names of matplotlib colors or 3-tuples of rgb values in range [0,255].
    data_range: list-like
        A list-like of the minimum and maximum values the data may take, respectively. Used to scale ``th``.
        Defaults to [0,1], for which a value of 0.5 in ``th`` would be the midpoint of the possible range.
    cmap_name: str
        The name of the colormap for matplotlib.
    """
    # Normalize threshold values based on the data range.
    th = list(map(lambda val: (val - data_range[0])/(data_range[1] - data_range[0]), th))
    # Normalize color values.
    for i, color in enumerate(colors):
        if isinstance(color, tuple):
            colors[i] = [rgb/255 for rgb in color]
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

## Misc ##

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

def skip_plot(n_pts, plot_type, kwargs={}):
    """Returns a boolean denoting whether to skip plotting data given the number of points it contains."""
    min_pts_dict = {'scatter': 1, 'box': 1, 'gaussian': 3, 'poly': 1, 'cubic_spline': 3, 'line':2}
    min_pts = min_pts_dict[plot_type]
    if plot_type == 'poly':
        assert 'degree' in kwargs.keys(), "When plotting a polynomal fit, there must be" \
                                              "a 'degree' entry in the fit_kwargs parameter."
        degree = kwargs['degree']
        min_pts = min_pts + degree
    return n_pts < min_pts

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