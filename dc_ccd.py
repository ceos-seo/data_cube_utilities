import ccd
from datetime import datetime, timedelta
from functools import reduce, partial
import itertools as it
import logging
import multiprocessing
import numpy as np
from operator import is_not

import warnings
import xarray

############################################################################
## Auxilary Functions
############################################################################

###### Time FUNCTIONS #################################


def _n64_to_int(n64):
    b = n64.astype(object)
    return int(b / 1000000000)


def _n64_to_datetime(n64):
    b = n64.astype(object)
    return datetime.fromtimestamp(int(b / 1000000000))


def _dt_to_sec(t):
    return datetime.fromtimestamp(t * 60 * 60 * 24)


###### Per Pixel FUNCTIONS ############################


def _run_ccd_on_pixel(ds):
    if 'time' not in ds.dims:
        raise Exception("You're missing time dims!")

    available_bands = ds.data_vars
    scene_count = ds.dims['time']

    date = [_n64_to_int(t) / 60 / 60 / 24 for t in ds.time.values]

    red = np.ones(scene_count) if 'red' not in available_bands else ds.red.values
    green = np.ones(scene_count) if 'green' not in available_bands else ds.green.values
    blue = np.ones(scene_count) if 'blue' not in available_bands else ds.blue.values
    nir = np.ones(scene_count) if 'nir' not in available_bands else ds.nir.values
    swir1 = np.ones(scene_count) if 'swir1' not in available_bands else ds.swir1.values
    swir2 = np.ones(scene_count) if 'swir2' not in available_bands else ds.swir2.values

    thermals = np.ones(scene_count) * (273.15) * 10 if 'thermal' not in available_bands else ds.object.values
    qa = np.array(ds.cf_mask.values)

    params = (date, blue, green, red, nir, swir1, swir2, thermals, qa)

    return ccd.detect(*params)


def _convert_ccd_results_into_dataset(results=None, model_dataset=None):

    start_times = [_dt_to_sec(model.start_day) for model in results['change_models']]

    intermediate_product = model_dataset.sel(time=start_times, method='nearest')

    new_dataset = xarray.DataArray(
        np.ones((intermediate_product.dims['time'], 1, 1)).astype(np.int16),
        coords=[
            intermediate_product.time.values, [intermediate_product.latitude.values],
            [intermediate_product.longitude.values]
        ],
        dims=['time', 'latitude', 'longitude'])

    return new_dataset.rename("continuous_change")


def is_pixel(value):
    return (len(value.latitude.dims) == 0) and (len(value.longitude.dims) == 0)


def clean_pixel(_ds, saturation_threshold=10000):
    # Filter out over saturated values
    ds = _ds
    mask = (ds < saturation_threshold) & (ds >= 0)
    indices = [x for x, y in enumerate(mask.red.values) if y == True]
    return ds.isel(time=indices)


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


def _save_plot_to_file(plot=None, file=None, band_name=None):
    if isinstance(file_name, str):
        file_name = [file_name]
    for fn in file_name:
        plot.savefig(str.replace(fn, "$BAND$", band), orientation='landscape', papertype='letter', bbox_inches='tight')


def _plot_band(results=None, original_pixel=None, band=None, file_name=None):
    fig = plt.figure(1)
    fig.suptitle(band.title(), fontsize=18, verticalalignment='bottom')

    lastdt = None

    dateLabels = []

    for change_model in results["change_models"]:
        target = getattr(change_model, band)

        time = np.arange(change_model.start_day, change_model.end_day, 1)

        ax1 = fig.add_subplot(211)

        xy = [(t, _lasso_eval(date=t, weights=target.coefficients, bias=target.intercept)) for t in time]
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
        xy = [(_n64_to_datetime(x.time.values) + timedelta(0), x.values) for x in clean_pixel(original_pixel)[band]
              if x < 5000]
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
        _save_plot_to_file(plot=plt, file=filename, band_name=band)

    plt.show()


##### Logging Decorators #################################################


def disable_logger(function):

    def _func(*params, **kwargs):
        logging.getLogger("ccd").setLevel(logging.WARNING)
        logging.getLogger("lcmap-pyccd").setLevel(logging.WARNING)
        result = function(*params, **kwargs)
        return result

    return _func


def enable_logger(function):

    def _func(*params, **kwargs):
        logging.getLogger("ccd").setLevel(logging.DEBUG)
        logging.getLogger("lcmap-pyccd").setLevel(logging.DEBUG)
        result = function(*params, **kwargs)
        return result

    return _func


##### THREAD OPS #################################################


def generate_thread_pool():
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2
    return multiprocessing.Pool(processes=(cpus))


def destroy_thread_pool(_pool):
    _pool.close()
    _pool.join()


###### ITERATOR FUNCTIONS ##########################################


def pixel_iterator_from_xarray(ds):
    lat_size = len(ds.latitude)
    lon_size = len(ds.longitude)
    cartesian = it.product(range(lat_size), range(lon_size))
    return map(lambda x: ds.isel(latitude=x[0], longitude=x[1]), cartesian)


def ccd_product_from_pixel(pixel):
    try:
        ccd_results = _run_ccd_on_pixel(pixel)
        ccd_product = _convert_ccd_results_into_dataset(results=ccd_results, model_dataset=pixel)
        return ccd_product
    except np.linalg.LinAlgError:
        # This is used to combat matrix inversion issues for Singular matrices.
        return None


def ccd_product_iterator_from_pixels(_pixels, distributed=False):
    if distributed == True:
        pool = generate_thread_pool()
        try:
            ccd_product_pixels = pool.imap_unordered(ccd_product_from_pixel, _pixels)
            destroy_thread_pool(pool)
            return ccd_product_pixels
        except:
            destroy_thread_pool(pool)
            raise
    else:
        ccd_product_pixels = map(ccd_product_from_pixel, _pixels)
        return ccd_product_pixels


def rebuild_xarray_from_pixels(_pixels):
    return reduce(lambda x, y: x.combine_first(y), _pixels)


###################################################################
## Callable Functions
###################################################################


@disable_logger
def process_xarray(ds, distributed=False):

    pixels = pixel_iterator_from_xarray(ds)
    ccd_products = ccd_product_iterator_from_pixels(pixels, distributed=distributed)
    ccd_products = filter(partial(is_not, None), ccd_products)
    ccd_change_count_xarray = rebuild_xarray_from_pixels(ccd_products)

    return (ccd_change_count_xarray.sum(dim='time') - 1).rename('change_volume')


@enable_logger
def process_pixel(ds):
    if is_pixel(ds) is not True:
        raise Exception("Incorrect dimensions for pixel operation.")

    duplicate_pixel = ds.copy(deep=True)
    ccd_results = _run_ccd_on_pixel(duplicate_pixel)

    duplicate_pixel.attrs['ccd_results'] = ccd_results

    duplicate_pixel.attrs['ccd_start_times'] = [_dt_to_sec(model.start_day) for model in ccd_results['change_models']]
    duplicate_pixel.attrs['ccd_end_times'] = [_dt_to_sec(model.end_day) for model in ccd_results['change_models']]
    duplicate_pixel.attrs['ccd_break_times'] = [_dt_to_sec(model.break_day) for model in ccd_results['change_models']]
    duplicate_pixel.attrs['ccd'] = True

    return duplicate_pixel


def plot_pixel(ds, bands=None):
    if 'ccd' not in list(ds.attrs.keys()):
        raise Exception("This pixel hasn't been processed by CCD. Use the `ccd.process_pixel()` function.")

    if bands is None or bands is []:
        possible_bands = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'thermal']
        avaliable_bands = ds.data_vars
        bands = intersect(possible_bands, avaliable_bands)

    for band in bands:
        _plot_band(results=ds.attrs['ccd_results'], original_pixel=ds, band=band)
