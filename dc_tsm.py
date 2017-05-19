import gc
import numpy as np
import xarray as xr
import scipy.ndimage.filters as conv

from . import dc_utilities as utilities
from datetime import datetime


def _tsmi(dataset):
    return (dataset.red.astype('float64') + dataset.green.astype('float64')) * 0.0001 / 2


def tsm(dataset_in, clean_mask=None, no_data=0):
    assert clean_mask is not None, "Please supply a boolean clean mask with the same dimensions as dataset_in"

    tsm = 3983 * _tsmi(dataset_in)**1.6246
    tsm.values[np.invert(clean_mask)] = no_data  # Contains data for clear pixels

    # Create xarray of data
    time = dataset_in.time
    latitude = dataset_in.latitude
    longitude = dataset_in.longitude
    dataset_out = xr.Dataset({'tsm': tsm}, coords={'time': time, 'latitude': latitude, 'longitude': longitude})
    return dataset_out


def mask_tsm(dataset_in, wofs):
    wofs_criteria = wofs.where(wofs > 0.8)
    wofs_criteria.values[wofs_criteria.values > 0] = 0
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    mask = conv.convolve(wofs_criteria.values, kernel, mode='constant')
    mask = mask.astype(np.float32)

    dataset_out = dataset_in.copy(deep=True)
    dataset_out.normalized_data.values += mask
    dataset_out.total_clean.values += mask
    utilities.nan_to_num(dataset_out, 0)

    return dataset_out
