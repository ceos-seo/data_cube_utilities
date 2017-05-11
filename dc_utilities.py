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

import gdal, osr
import numpy as np
import xarray as xr
import collections
import os
import math
import datetime
import shutil
import uuid
import rasterio

import datacube

# Author: KMF
# Creation date: 2016-06-13
"""
General-use functions
"""


def create_cfmask_clean_mask(cfmask, no_data=-9999):
    """
    Description:
      Create a clean mask for clear land/water pixels,
      i.e. mask out shadow, snow, cloud, and no data
    -----
    Input:
      cfmask (xarray) - cf_mask from the ledaps products
    Output:
      clean_mask (boolean numpy array) - clear land/water mask
    """

    #########################
    # cfmask values:        #
    #   0 - clear           #
    #   1 - water           #
    #   2 - cloud shadow    #
    #   3 - snow            #
    #   4 - cloud           #
    #   255 - fill          #
    #########################

    clean_mask = np.reshape(
        np.in1d(cfmask.values.reshape(-1), [2, 3, 4, 255, no_data], invert=True), cfmask.values.shape)
    return clean_mask


# split a task (sq area, time) into geographical and time chunks based on params.
# latitude and longitude are a tuple containing (lower, upper)
# acquisitions are the list of all acquisitions
# geo chunk size is the square area per chunk, time chunks is the number of time chunks.
# returns list of ranges that make up the full lat/lon ranges, list of lists containing acquisition dates.
def split_task(resolution=0.000269,
               latitude=None,
               longitude=None,
               acquisitions=None,
               geo_chunk_size=None,
               time_chunks=None,
               reverse_time=False):
    square_area = (longitude[1] - longitude[0]) * (latitude[1] - latitude[0])
    print("Square area: ", square_area)
    #split the task into n geo chunks based on sq area and chunk size.
    # checking if lat/lon are not none as its possible to not enter params and get full dataset.
    lon_ranges = []
    lat_ranges = []
    if latitude is not None and longitude is not None:
        if geo_chunk_size is not None and square_area > geo_chunk_size:
            geographic_chunks = math.ceil(square_area / geo_chunk_size)
            lat_range_size = (latitude[1] - latitude[0]) / geographic_chunks
            # longitude/x
            for i in range(math.ceil((latitude[1] - latitude[0]) / lat_range_size)):
                lower_lat = latitude[0] + (i * lat_range_size)
                upper_lat = latitude[0] + ((i + 1) * lat_range_size)
                if i != geographic_chunks - 1:
                    upper_lat -= resolution
                lat_ranges.append((lower_lat, upper_lat))
                lon_ranges.append((longitude[0], longitude[1]))
        else:
            lon_ranges.append((longitude[0], longitude[1]))
            lat_ranges.append((latitude[0], latitude[1]))
    else:
        lon_ranges = [None]
        lat_ranges = [None]
    #split the acquisition list into chunks as well.
    acquisitions_sorted = sorted(acquisitions)
    time_ranges = [list(reversed(acquisitions_sorted)) if reverse_time else acquisitions_sorted]
    if time_chunks is not None:
        time_chunk_size = math.ceil(len(acquisitions_sorted) / time_chunks)
        time_ranges = list(
            chunks(list(reversed(acquisitions_sorted)) if reverse_time else acquisitions_sorted, time_chunk_size))

    return lat_ranges, lon_ranges, time_ranges


def generate_time_ranges(acquisition_list, reverse_time, slices_per_iteration):
    time_index = 0
    while time_index < len(acquisition_list):
        start = acquisition_list[time_index] + datetime.timedelta(
            seconds=1) if reverse_time else acquisition_list[time_index]
        if slices_per_iteration is not None and (time_index + slices_per_iteration - 1) < len(acquisition_list):
            end = acquisition_list[time_index + slices_per_iteration - 1]
        else:
            end = acquisition_list[-1] if reverse_time else acquisition_list[-1] + datetime.timedelta(seconds=1)
        time_range = tuple(sorted((start, end)))
        yield time_range
        time_index = time_index + (slices_per_iteration if slices_per_iteration is not None else len(acquisition_list))


def get_spatial_ref(crs):
    """
    Description:
      Get the spatial reference of a given crs
    -----
    Input:
      crs (datacube.model.CRS) - Example: CRS('EPSG:4326')
    Output:
      ref (str) - spatial reference of given crs
    """

    crs_str = str(crs)
    epsg_code = int(crs_str.split(':')[1])
    ref = osr.SpatialReference()
    ref.ImportFromEPSG(epsg_code)
    return str(ref)


def perform_timeseries_analysis(dataset_in, no_data=-9999):
    """
    Description:

    -----
    Input:
      dataset_in (xarray.DataSet) - dataset with one variable to perform timeseries on
    Output:
      dataset_out (xarray.DataSet) - dataset containing
        variables: normalized_data, total_data, total_clean
    """

    data_vars = list(dataset_in.data_vars)
    key = data_vars[0]
    data = dataset_in[key].astype('float')

    processed_data = data.copy(deep=True)
    processed_data.values[data.values == no_data] = 0
    processed_data_sum = processed_data.sum('time')

    clean_data = data.copy(deep=True)
    clean_data.values[data.values != no_data] = 1
    clean_data.values[data.values == no_data] = 0
    clean_data_sum = clean_data.sum('time')

    processed_data_normalized = processed_data_sum / clean_data_sum

    dataset_out = xr.Dataset(
        {
            'normalized_data': processed_data_normalized,
            'total_data': processed_data_sum,
            'total_clean': clean_data_sum
        },
        coords={'latitude': dataset_in.latitude,
                'longitude': dataset_in.longitude})

    return dataset_out


def perform_timeseries_analysis_iterative(dataset_in, intermediate_product=None, no_data=-9999):
    """
    Description:

    -----
    Input:
      dataset_in (xarray.DataSet) - dataset with one variable to perform timeseries on
    Output:
      dataset_out (xarray.DataSet) - dataset containing
        variables: normalized_data, total_data, total_clean
    """

    data_vars = list(dataset_in.data_vars)
    key = data_vars[0]
    data = dataset_in[key].astype('float')

    processed_data = data.copy(deep=True)
    processed_data.values[data.values == no_data] = 0
    processed_data_sum = processed_data.sum('time')

    clean_data = data.copy(deep=True)
    clean_data.values[data.values != no_data] = 1
    clean_data.values[data.values == no_data] = 0
    clean_data_sum = clean_data.sum('time')

    if intermediate_product is None:
        processed_data_normalized = processed_data_sum / clean_data_sum
        processed_data_normalized.values[np.isnan(processed_data_normalized.values)] = 0
        dataset_out = xr.Dataset(
            {
                'normalized_data': processed_data_normalized,
                'total_data': processed_data_sum,
                'total_clean': clean_data_sum
            },
            coords={'latitude': dataset_in.latitude,
                    'longitude': dataset_in.longitude})
    else:
        dataset_out = intermediate_product.copy(deep=True)
        dataset_out['total_data'] += processed_data_sum
        dataset_out['total_clean'] += clean_data_sum
        processed_data_normalized = dataset_out['total_data'] / dataset_out['total_clean']
        processed_data_normalized.values[np.isnan(processed_data_normalized.values)] = 0
        dataset_out['normalized_data'] = processed_data_normalized

    return dataset_out


"""

AHDS: class view refactor

"""


def clear_attrs(dataset):
    """Clear out all attributes on an xarray dataset to write to disk."""
    dataset.attrs = collections.OrderedDict()
    for band in dataset:
        dataset[band].attrs = collections.OrderedDict()


def create_bit_mask(data_array, valid_bits, no_data=-9999):
    """Create a boolean bit mask from a list of valid bits

    Args:
        data_array: xarray data array to extract bit information for.
        valid_bits: array of ints representing what bits should be considered valid.
        nodata: nodata value for the data array.

    Returns:
        Boolean mask signifying valid data.

    """
    assert isinstance(valid_bits, list) and isinstance(valid_bits[0], int), "Valid bits must be a list of integer bits"
    #do bitwise and on valid mask - all zeros means no intersection e.g. invalid else return a truthy value?
    valid_mask = sum([1 >> valid_bit for bit in valid_bits])
    clean_mask = (data_array.values & valid_mask).astype('bool')
    return clean_mask


def add_timestamp_data_to_xr(dataset):
    """Add timestamp data to an xarray dataset using the time dimension.

    Adds both a timestamp and a human readable date int to a dataset - int32 format required.
    modifies the dataset in place.
    """
    dims_data_var = list(dataset.data_vars)[0]

    timestamp_data = np.full(dataset[dims_data_var].values.shape, 0, dtype="int32")
    date_data = np.full(dataset[dims_data_var].values.shape, 0, dtype="int32")

    for index, acq_date in enumerate(dataset.time.values.astype('M8[ms]').tolist()):
        timestamp_data[index::] = acq_date.timestamp()
        date_data[index::] = int(acq_date.strftime("%Y%m%d"))
    dataset['timestamp'] = xr.DataArray(
        timestamp_data,
        dims=('time', 'latitude', 'longitude'),
        coords={'latitude': dataset.latitude,
                'longitude': dataset.longitude,
                'time': dataset.time})
    dataset['date'] = xr.DataArray(
        date_data,
        dims=('time', 'latitude', 'longitude'),
        coords={'latitude': dataset.latitude,
                'longitude': dataset.longitude,
                'time': dataset.time})


def write_geotiff_from_xr(tif_path, dataset, bands, nodata=-9999, crs="EPSG:4326"):
    """Write a geotiff from an xarray dataset.

    Args:
        tif_path: path for the tif to be written to.
        dataset: xarray dataset
        bands: list of strings representing the bands in the order they should be written
        nodata: nodata value for the dataset
        crs: requested crs.

    """
    assert isinstance(bands, list), "Bands must a list of strings"
    assert len(bands) > 0 and isinstance(bands[0], str), "You must supply at least one band."
    with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=dataset.dims['latitude'],
            width=dataset.dims['longitude'],
            count=len(bands),
            dtype=str(dataset[bands[0]].dtype),
            crs=crs,
            transform=_get_transform_from_xr(dataset),
            nodata=nodata) as dst:
        for index, band in enumerate(bands):
            dst.write(dataset[band].values, index + 1)
        dst.close()


def write_png_from_xr(png_path, dataset, bands, png_filled_path=None, fill_color='red', scale=None):
    """Write a rgb png from an xarray dataset.

    Args:
        png_path: path for the png to be written to.
        dataset: dataset to use for the png creation.
        bands: a list of three strings representing the bands and their order
        png_filled_path: optional png with nodata values filled
        fill_color: color to use as the nodata fill
        scale: desired scale - tuple like (0, 4000) for the upper and lower bounds

    """
    assert isinstance(bands, list), "Bands must a list of strings"
    assert len(bands) == 3 and isinstance(bands[0], str), "You must supply three string bands for a PNG."

    tif_path = os.path.join(os.path.dirname(png_path), str(uuid.uuid4()) + ".png")
    write_geotiff_from_xr(tif_path, dataset, bands)

    scale_string = "-scale " + str(scale[0]) + " " + str(scale[1]) + " 0 255" if scale is not None else ""
    cmd = "gdal_translate -ot Byte -outsize 25% 25% " + scale_string + " -of PNG -b 1 -b 2 -b 3 " + tif_path + ' ' + png_path

    os.system(cmd)

    if png_filled_path is not None and fill_color is not None:
        cmd = "convert -transparent \"#000000\" " + png_path + " " + png_path
        os.system(cmd)
        cmd = "convert " + png_path + " -background " + \
            fill_color + " -alpha remove " + png_filled_path
        os.system(cmd)

    os.remove(tif_path)


def write_single_band_png_from_xr(png_path, dataset, band, color_scale=None, fill_color=None):
    """Write a pseudocolor png from an xarray dataset.

    Args:
        png_path: path for the png to be written to.
        dataset: dataset to use for the png creation.
        band: The band to write to a png
        png_filled_path: optional png with nodata values filled
        fill_color: color to use as the nodata fill
        color_scale: path to a color scale compatible with gdal.

    """
    assert os.path.exists(color_scale), "Color scale must be a path to a text file containing a gdal compatible scale."
    assert isinstance(bands, str), "Band must be a string."

    tif_path = os.path.join(os.path.dirname(png_path), str(uuid.uuid4()) + ".png")
    write_geotiff_from_xr(tif_path, dataset, [band])

    cmd = "gdaldem color-relief -of PNG -b 1 " + tif_path + " " + \
        color_scale + " " + png_path
    os.system(cmd)
    cmd = "convert -transparent \"#FFFFFF\" " + \
        png_path + " " + png_path
    os.system(cmd)
    if fill_color is not None and fill_color != "transparent":
        cmd = "convert " + png_path + " -background " + \
            fill_color + " -alpha remove " + png_path
        os.system(cmd)

    os.remove(tif_path)


def _get_transform_from_xr(dataset):
    """Create a geotransform from an xarray dataset.
    """

    from rasterio.transform import from_bounds
    geotransform = from_bounds(dataset.longitude[0], dataset.latitude[-1], dataset.longitude[-1], dataset.latitude[0],
                               len(dataset.longitude), len(dataset.latitude))
    return geotransform


"""

END

"""


def save_to_geotiff(out_file,
                    data_type,
                    dataset_in,
                    geotransform,
                    spatial_ref,
                    x_pixels=3711,
                    y_pixels=3712,
                    no_data=-9999,
                    band_order=None):
    """
    Description:
      Save data in bands to a GeoTIFF
    -----
    Inputs:
      out_file (str) - name of output file
      data_type (gdal data type) - gdal.GDT_Int16, gdal.GDT_Float32, etc
      dataset_in (xarray dataset) - xarray dataset containing only bands to output.
      geotransform (tuple) - (ul_lon, lon_dist, lon_rtn, ul_lat, lat_rtn, lat_dist)
      spatial_ref (str) - spatial reference of dataset's crs
    Optional Inputs:
      x_pixels (int) - num pixels in x direction
      y_pixels (int) - num pixels in y direction
      no_data (int/float) - no data value
      band_order - list of bands in order for the tiff.
    """

    data_vars = dataset_in.data_vars

    if band_order is None:
        keys = list(data_vars)
    else:
        keys = band_order

    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(
        out_file, x_pixels, y_pixels, len(keys), data_type, options=["BIGTIFF=YES", "INTERLEAVE=BAND"])
    raster.SetGeoTransform(geotransform)
    raster.SetProjection(spatial_ref)
    index = 1
    for key in keys:
        out_band = raster.GetRasterBand(index)
        out_band.SetNoDataValue(no_data)
        out_band.WriteArray(data_vars[key].values)
        out_band.FlushCache()
        index += 1
    raster.FlushCache()
    out_band = None
    raster = None


def create_rgb_png_from_tiff(tif_path,
                             png_path,
                             bands=[1, 2, 3],
                             png_filled_path=None,
                             fill_color=None,
                             scale=None,
                             per_band_scale=False):
    scale_string = ""
    if scale is not None:
        if per_band_scale:
            for index, band in enumerate(bands):
                scale_string += " -scale_" + str(band) + " " + str(scale[index][0]) + " " + str(
                    scale[index][1]) + " 0 255"
        else:
            scale_string += "-scale " + str(scale[0]) + " " + str(scale[1]) + " 0 255"
    cmd = "gdal_translate -ot Byte -outsize 25% 25% " + scale_string + " -of PNG -b " + str(bands[0]) + " -b " + str(bands[1]) + " -b " + str(bands[2]) + " " + \
        tif_path + ' ' + png_path

    os.system(cmd)

    if png_filled_path is not None and fill_color is not None:
        cmd = "convert -transparent \"#000000\" " + png_path + " " + png_path
        os.system(cmd)
        cmd = "convert " + png_path + " -background " + \
            fill_color + " -alpha remove " + png_filled_path
        os.system(cmd)


def create_single_band_rgb(band=None, tif_path=None, color_scale=None, output_path=None, fill=None):
    cmd = "gdaldem color-relief -of PNG -b " + \
        str(band) + " " + tif_path + " " + \
        color_scale + " " + output_path
    os.system(cmd)
    cmd = "convert -transparent \"#FFFFFF\" " + \
        output_path + " " + output_path
    os.system(cmd)
    if fill is not None and fill != "transparent":
        cmd = "convert " + output_path + " -background " + \
            fill + " -alpha remove " + output_path
        os.system(cmd)


# Break the list l into n sized chunks.
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


"""
functions used to combine time sliced data after being combined geographically.
Fill nodata uses the first timeslice as a base, then uses subsequent slices to
fill in indices with nodata values.
this should be used for recent/leastrecent + anything that is done in a single time chunk (median pixel?)
things like max/min ndvi should be able to compound max/min ops between ddifferent timeslices so this will be
different for that.
"""


def fill_nodata(dataset, dataset_intermediate):
    if dataset_intermediate is None:
        return dataset.copy(deep=True)
    dataset_out = dataset_intermediate.copy(deep=True)
    for key in list(dataset_out.data_vars):
        # Get raw data for current variable and mask the data
        dataset_out[key].values[dataset_out[key].values == -9999] = dataset[key].values[dataset_out[key].values ==
                                                                                        -9999]
    return dataset_out


def max_value(dataset, dataset_intermediate):
    if dataset_intermediate is None:
        return dataset.copy(deep=True)
    dataset_out = dataset_intermediate.copy(deep=True)
    for key in list(dataset_out.data_vars):
        # Get raw data for current variable and mask the data
        dataset_out[key].values[dataset.ndvi.values >
                                dataset_out.ndvi.values] = dataset[key].values[dataset.ndvi.values >
                                                                               dataset_out.ndvi.values]
    return dataset_out


def min_value(dataset, dataset_intermediate):
    if dataset_intermediate is None:
        return dataset.copy(deep=True)
    dataset_out = dataset_intermediate.copy(deep=True)
    for key in list(dataset_out.data_vars):
        # Get raw data for current variable and mask the data
        dataset_out[key].values[dataset.ndvi.values <
                                dataset_out.ndvi.values] = dataset[key].values[dataset.ndvi.values <
                                                                               dataset_out.ndvi.values]
    return dataset_out


def addition(dataset, dataset_intermediate):
    """
    functions used to combine time sliced data after being combined geographically.
    This compounds the results of the time slice and recomputes the normalized data.
    """
    if dataset_intermediate is None:
        return dataset.copy(deep=True)
    data_vars = ["total_data", "total_clean"]
    dataset_out = dataset_intermediate.copy(deep=True)
    for key in data_vars:
        dataset_out[key].values += dataset[key].values
    dataset_out['normalized_data'].values = dataset_out["total_data"].values / dataset_out["total_clean"].values
    dataset_out['normalized_data'].values[np.isnan(dataset_out['normalized_data'].values)] = 0
    return dataset_out
