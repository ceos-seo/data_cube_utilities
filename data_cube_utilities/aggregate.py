import pandas as pd

def xarray_bin_time(dataset, num_bins):
    """
    Take the mean of an xarray along time with binning. 
    All dimensions other than time are collapsed via mean for each bin.
    Useful for plotting data with high variance in temporal spacing between acquisitions.
    
    Parameters
    ----------
    dataset: xarray.Dataset or xarray.DataArray
        The Dataset or DataArray to aggregate by binning.
        Must have a 'time' coordinate of type `datetime64`.
    num_bins: int
        The number of bins to use.
    
    Returns
    -------
    result: xarray.Dataset or xarray.DataArray
        The result of aggregating within bins for the binned data.
    """
    start_time, end_time = map(lambda time: pd.to_datetime(time), dataset.time.values[[0,-1]])
    total_days = (end_time-start_time).days
    bins = list(pd.interval_range(start=start_time, end=end_time, freq="{}D".format(int(total_days/num_bins))))
    for i, intval in enumerate(bins):
        bins[i] = intval.left + (intval.right - intval.left)/2
    agg_time = list(map(lambda pd_ts: pd_ts.to_datetime64(), bins))
    result = dataset.interp(time=agg_time)
    return result
