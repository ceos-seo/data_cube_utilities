import fiona
import xarray as xr
import numpy as np

from rasterio.features import geometry_mask
from shapely.geometry import shape


def shapefile_mask(dataset: xr.Dataset, shapefile) -> np.array:
    """Extracts a mask from a shapefile using dataset latitude and longitude extents.

    Args:
        dataset (xarray.Dataset): The dataset with the latitude and longitude extents.
        shapefile (string): The shapefile to be used for extraction.

    Returns:
        A boolean mask array.
    """
    with fiona.open(shapefile, 'r') as source:
        collection = list(source)
        geometries = []
        for feature in collection:
            geometries.append(shape(feature['geometry']))
        geobox = dataset.geobox
        mask = geometry_mask(
            geometries,
            out_shape=geobox.shape,
            transform=geobox.affine,
            all_touched=True,
            invert=True)
    return mask