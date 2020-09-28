# Copyright 2016 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Portion of this code is Copyright Geoscience Australia, Licensed under the
# Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License
# at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# The CEOS 2 platform is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import os
# old_cwd = os.getcwd()
# os.chdir(os.path.dirname(__file__))

import gc
import numpy as np
import xarray as xr
import dask
import datacube
import warnings

# Command line tool imports
import argparse
import collections
import gdal
from datetime import datetime


from . import dc_utilities as utilities
from .dc_utilities import create_default_clean_mask

# os.chdir(old_cwd)

# Author: KMF
# Creation date: 2016-06-13

def NDWI(data, normalize=False, band_pair=0):
    """
    Computes various versions of the Normalized Difference Water Index for an `xarray.Dataset`.
    Values should be in the range [-1,1] for valid LANDSAT data (the bands are positive).

    Parameters
    ----------
    data: xarray.Dataset or numpy.ndarray
        An `xarray.Dataset` containing the bands specified by `band_pair` or
        a 2D NumPy array with two columns - the band pair.
    normalize: bool
        Whether or not to normalize to the range [0,1].
    band_pair: int
        The band pair to use.
        Band pair 0 uses 'nir' and 'swir1': (nir - swir1)/(nir + swir1).
        Band pair 1 uses 'green' and 'nir': (green - nir)/(green + nir).

    Returns
    -------
    ndwi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `dataset` - the same coordinates in
        the same order.
    """
    bands = [None] * 2
    if band_pair == 0:
        bands = ['nir', 'swir1']
    elif band_pair == 1:
        bands = ['green', 'nir']
    else:
        raise AssertionError('The band_pair parameter must be in [0,1]')

    if isinstance(data, xr.Dataset):
        ndwi = (data[bands[0]] - data[bands[1]]) / (data[bands[0]] + data[bands[1]])
        if normalize:
            ndwi = (ndwi - ndwi.min())/(ndwi.max() - ndwi.min())
    else:
        ndwi = data[:,0] - data[:,1]
        if normalize:
            ndwi = (ndwi - np.nanmin(ndwi))/(np.nanmax(ndwi) - np.nanmin(ndwi))
    return ndwi

def wofs_classify(dataset_in, clean_mask=None, x_coord='longitude', y_coord='latitude',
                  time_coord='time', no_data=-9999, mosaic=False):
    """
    Description:
      Performs WOfS algorithm on given dataset.
    Assumption:
      - The WOfS algorithm is defined for Landsat 5/Landsat 7
    References:
      - Mueller, et al. (2015) "Water observations from space: Mapping surface water from
        25 years of Landsat imagery across Australia." Remote Sensing of Environment.
      - https://github.com/GeoscienceAustralia/eo-tools/blob/stable/eotools/water_classifier.py
    -----
    Inputs:
      dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube; should contain
        coordinates: time, latitude, longitude
        variables: blue, green, red, nir, swir1, swir2
    x_coord, y_coord, time_coord: (str) - Names of DataArrays in `dataset_in` to use as x, y,
        and time coordinates.
    Optional Inputs:
      clean_mask (nd numpy array with dtype boolean) - true for values user considers clean;
        if user does not provide a clean mask, all values will be considered clean
      no_data (int/float) - no data pixel value; default: -9999
      mosaic (boolean) - flag to indicate if dataset_in is a mosaic. If mosaic = False, dataset_in
        should have a time coordinate and wofs will run over each time slice; otherwise, dataset_in
        should not have a time coordinate and wofs will run over the single mosaicked image
    Output:
      dataset_out (xarray.DataArray) - wofs water classification results: 0 - not water; 1 - water
    Throws:
        ValueError - if dataset_in is an empty xarray.Dataset.
    """

    def _band_ratio(a, b):
        """
        Calculates a normalized ratio index
        """
        return (a - b) / (a + b)

    def _run_regression(band1, band2, band3, band4, band5, band7):
        """
        Regression analysis based on Australia's training data
        TODO: Return type
        """

        # Compute normalized ratio indices
        ndi_52 = _band_ratio(band5, band2)
        ndi_43 = _band_ratio(band4, band3)
        ndi_72 = _band_ratio(band7, band2)

        if isinstance(band1, np.ndarray):
            classified = np.full_like(band1, no_data, dtype='uint8')
        elif isinstance(band1, dask.array.core.Array):
            classified = dask.array.full_like(band1, no_data, dtype='uint8',
                                    chunks=band1.chunks)
    
        # Start with the tree's left branch, finishing nodes as needed

        # Left branch
        r1 = ndi_52 <= -0.01

        r2 = band1 <= 2083.5
        classified[r1 & ~r2] = 0  #Node 3

        r3 = band7 <= 323.5
        _tmp = r1 & r2
        _tmp2 = _tmp & r3
        _tmp &= ~r3

        r4 = ndi_43 <= 0.61
        classified[_tmp2 & r4] = 1  #Node 6
        classified[_tmp2 & ~r4] = 0  #Node 7

        r5 = band1 <= 1400.5
        _tmp2 = _tmp & ~r5

        r6 = ndi_43 <= -0.01
        classified[_tmp2 & r6] = 1  #Node 10
        classified[_tmp2 & ~r6] = 0  #Node 11

        _tmp &= r5

        r7 = ndi_72 <= -0.23
        _tmp2 = _tmp & ~r7

        r8 = band1 <= 379
        classified[_tmp2 & r8] = 1  #Node 14
        classified[_tmp2 & ~r8] = 0  #Node 15

        _tmp &= r7

        r9 = ndi_43 <= 0.22
        classified[_tmp & r9] = 1  #Node 17
        _tmp &= ~r9

        r10 = band1 <= 473
        classified[_tmp & r10] = 1  #Node 19
        classified[_tmp & ~r10] = 0  #Node 20

        # Left branch complete; cleanup
        del r2, r3, r4, r5, r6, r7, r8, r9, r10
        gc.collect()

        # Right branch of regression tree
        r1 = ~r1

        r11 = ndi_52 <= 0.23
        _tmp = r1 & r11

        r12 = band1 <= 334.5
        _tmp2 = _tmp & ~r12
        classified[_tmp2] = 0  #Node 23

        _tmp &= r12

        r13 = ndi_43 <= 0.54
        _tmp2 = _tmp & ~r13
        classified[_tmp2] = 0  #Node 25

        _tmp &= r13

        r14 = ndi_52 <= 0.12
        _tmp2 = _tmp & r14
        classified[_tmp2] = 1  #Node 27

        _tmp &= ~r14

        r15 = band3 <= 364.5
        _tmp2 = _tmp & r15

        r16 = band1 <= 129.5
        classified[_tmp2 & r16] = 1  #Node 31
        classified[_tmp2 & ~r16] = 0  #Node 32

        _tmp &= ~r15

        r17 = band1 <= 300.5
        _tmp2 = _tmp & ~r17
        _tmp &= r17
        classified[_tmp] = 1  #Node 33
        classified[_tmp2] = 0  #Node 34

        _tmp = r1 & ~r11

        r18 = ndi_52 <= 0.34
        classified[_tmp & ~r18] = 0  #Node 36
        _tmp &= r18

        r19 = band1 <= 249.5
        classified[_tmp & ~r19] = 0  #Node 38
        _tmp &= r19

        r20 = ndi_43 <= 0.45
        classified[_tmp & ~r20] = 0  #Node 40
        _tmp &= r20

        r21 = band3 <= 364.5
        classified[_tmp & ~r21] = 0  #Node 42
        _tmp &= r21

        r22 = band1 <= 129.5
        classified[_tmp & r22] = 1  #Node 44
        classified[_tmp & ~r22] = 0  #Node 45

        # Completed regression tree

        return classified

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)
    
    # Extract dataset bands needed for calculations
    blue = dataset_in.blue
    green = dataset_in.green
    red = dataset_in.red
    nir = dataset_in.nir
    swir1 = dataset_in.swir1
    swir2 = dataset_in.swir2

    # Ignore warnings about division by zero and NaNs.
    classified = _run_regression(blue.data, green.data, red.data, 
                                 nir.data, swir1.data, swir2.data)
     
    # classified_clean = classified - classified + no_data
    
    if isinstance(blue.data, np.ndarray):
        classified_clean = np.where(clean_mask, classified, no_data)
    elif isinstance(blue.data, dask.array.core.Array):
        classified_clean = dask.array.where(clean_mask, classified, no_data)
    
    # Create xarray of data
    x_coords = dataset_in[x_coord]
    y_coords = dataset_in[y_coord]

    time = None
    coords = None
    dims = None

    if mosaic:
        coords = [y_coords, x_coords]
        dims = [y_coord, x_coord]
    else:
        time_coords = dataset_in[time_coord]
        coords = [time_coords, y_coords, x_coords]
        dims = [time_coord, y_coord, x_coord]

    data_array = xr.DataArray(classified_clean, coords=coords, dims=dims)

    if mosaic:
        dataset_out = xr.Dataset({'wofs': data_array},
                                 coords={y_coord: y_coords, x_coord: x_coords})
    else:
        dataset_out = xr.Dataset(
            {'wofs': data_array},
            coords={time_coord: time_coords, y_coord: y_coords, x_coord: x_coords})

    return dataset_out


def ledaps_classify(water_band, qa_bands, no_data=-9999):
    #TODO: refactor for input/output datasets

    fill_qa = qa_bands[0]
    cloud_qa = qa_bands[1]
    cloud_shadow_qa = qa_bands[2]
    adjacent_cloud_qa = qa_bands[3]
    snow_qa = qa_bands[4]
    ddv_qa = qa_bands[5]

    fill_mask = np.reshape(np.in1d(fill_qa.reshape(-1), [0]), fill_qa.shape)
    cloud_mask = np.reshape(np.in1d(cloud_qa.reshape(-1), [0]), cloud_qa.shape)
    cloud_shadow_mask = np.reshape(np.in1d(cloud_shadow_qa.reshape(-1), [0]), cloud_shadow_qa.shape)
    adjacent_cloud_mask = np.reshape(np.in1d(adjacent_cloud_qa.reshape(-1), [255]), adjacent_cloud_qa.shape)
    snow_mask = np.reshape(np.in1d(snow_qa.reshape(-1), [0]), snow_qa.shape)
    ddv_mask = np.reshape(np.in1d(ddv_qa.reshape(-1), [0]), ddv_qa.shape)

    clean_mask = fill_mask & cloud_mask & cloud_shadow_mask & adjacent_cloud_mask & snow_mask & ddv_mask

    water_mask = np.reshape(np.in1d(water_band.reshape(-1), [255]), water_band.shape)  #Will be true if 255 -> water

    classified = np.copy(water_mask)
    classified.astype(int)

    classified_clean = np.full(classified.shape, no_data)
    classified_clean[clean_mask] = classified[clean_mask]

    return classified_clean


def cfmask_classify(cfmask, no_data=-9999):
    #TODO: refactor for input/output datasets

    clean_mask = np.reshape(np.in1d(cfmask.reshape(-1), [2, 3, 4, 255], invert=True), cfmask.shape)

    water_mask = np.reshape(np.in1d(cfmask.reshape(-1), [1]), cfmask.shape)

    classified = np.copy(water_mask)
    classified.astype(int)

    classified_clean = np.full(classified.shape, no_data)
    classified_clean[clean_mask] = classified[clean_mask]

    return classified_clean


def main(classifier, platform, product_type, min_lon, max_lon, min_lat, max_lat, start_date, end_date, dc_config):
    """
    Description:
      Command-line water detection tool - creates a time-series from
        water analysis performed on data retrieved by the Data Cube,
        shows plots of the normalized water observations (total water
        observations / total clear observations), total water observations,
        and total clear observations, and saves a GeoTIFF of the results
    Assumptions:
      The command-line tool assumes there is a measurement called cf_mask
    Inputs:
      classifier (str)
      platform (str)
      product_type (str)
      min_lon (str)
      max_lon (str)
      min_lat (str)
      max_lat (str)
      start_date (str)
      end_date (str)
      dc_config (str)
    """

    # Initialize data cube object
    dc = datacube.Datacube(config=dc_config, app='dc-mosaicker')

    # Validate arguments
    if classifier not in ['cfmask', 'ledaps', 'wofs']:
        print('ERROR: Unknown water classifier. Classifier options: cfmask, ledaps, wofs')
        return

    products = dc.list_products()
    platform_names = set([product[6] for product in products.values])
    if platform not in platform_names:
        print('ERROR: Invalid platform.')
        print('Valid platforms are:')
        for name in platform_names:
            print(name)
        return

    product_names = [product[0] for product in products.values]
    if product_type not in product_names:
        print('ERROR: Invalid product type.')
        print('Valid product types are:')
        for name in product_names:
            print(name)
        return

    try:
        min_lon = float(args.min_lon)
        max_lon = float(args.max_lon)
        min_lat = float(args.min_lat)
        max_lat = float(args.max_lat)
    except:
        print('ERROR: Longitudes/Latitudes must be float values')
        return

    try:
        start_date_str = start_date
        end_date_str = end_date
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    except:
        print('ERROR: Invalid date format. Date format: YYYY-MM-DD')
        return

    if not os.path.exists(dc_config):
        print('ERROR: Invalid file path for dc_config')
        return

    # Retrieve data from Data Cube
    dataset_in = dc.load(
        platform=platform,
        product=product_type,
        time=(start_date, end_date),
        lon=(min_lon, max_lon),
        lat=(min_lat, max_lat))

    # Get information needed for saving as GeoTIFF

    # Spatial ref
    crs = dataset_in.crs
    spatial_ref = utilities.get_spatial_ref(crs)

    # Upper left coordinates
    ul_lon = dataset_in.longitude.values[0]
    ul_lat = dataset_in.latitude.values[0]

    # Resolution
    products = dc.list_products()
    resolution = products.resolution[products.name == 'ls7_ledaps']
    lon_dist = resolution.values[0][1]
    lat_dist = resolution.values[0][0]

    # Rotation
    lon_rtn = 0
    lat_rtn = 0

    geotransform = (ul_lon, lon_dist, lon_rtn, ul_lat, lat_rtn, lat_dist)

    # Run desired classifier
    water_class = None
    if classifier == 'cfmask':  #TODO: implement when cfmask_classify is refactored
        return
    elif classifier == 'ledaps':  #TODO: implement when cfmask_classify is refactored
        return
    elif classifier == 'wofs':
        water_class = wofs_classify(dataset_in)

    dataset_out = utilities.perform_timeseries_analysis(water_class)

    print(dataset_out)

    out_file = (
        str(min_lon) + '_' + str(min_lat) + '_' + start_date_str + '_' + end_date_str + '_' + classifier + '_.tif')

    utilities.save_to_geotiff(out_file, gdal.GDT_Float32, dataset_out, geotransform, spatial_ref)


if __name__ == '__main__':

    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('classifier', help='Water classifier; options: cfmask, ledaps, wofs')
    parser.add_argument('platform', help='Data platform; example: LANDSAT_7')
    parser.add_argument('product', help='Product type; example: ls7_ledaps')
    parser.add_argument('min_lon', help='Minimum longitude')
    parser.add_argument('max_lon', help='Maximum longitude')
    parser.add_argument('min_lat', help='Minimum latitude')
    parser.add_argument('max_lat', help='Maximum latitude')
    parser.add_argument('start_date', help='Start date; format: YYYY-MM-DD')
    parser.add_argument('end_date', help='End date; format: YYYY-MM-DD')
    parser.add_argument(
        'dc_config',
        nargs='?',
        default='~/.datacube.conf',
        help='Datacube configuration path; default: ~/.datacube.conf')

    args = parser.parse_args()

    main(args.classifier, args.platform, args.product, args.min_lon, args.max_lon, args.min_lat, args.max_lat,
         args.start_date, args.end_date, args.dc_config)

    end_time = datetime.now()
    print('Execution time: ' + str(end_time - start_time))
