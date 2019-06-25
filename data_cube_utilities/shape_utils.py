import numpy as np

def shapely_geom_intersects_rect(geom, x, y):
    """
    Determines whether the bounding box of a Shapely polygon intersects
    a rectangle defined by `x` and `y` extents.

    Parameters
    ----------
    geom: shapely.geometry.polygon.Polygon
        The object to determine intersection with the region defined by `x` and `y`.
    x, y: list-like
        The x and y extents, expressed as 2-tuples.

    Returns
    -------
    intersects: bool
        Whether the bounding box of `geom` intersects the rectangle.
    """
    geom_bounds = np.array(list(geom.bounds))
    x_shp, y_shp = geom_bounds[[0, 2]], geom_bounds[[1, 3]]
    x_in_range = (x_shp[0] < x[1]) & (x[0] < x_shp[1])
    y_in_range = (y_shp[0] < y[1]) & (y[0] < y_shp[1])
    return x_in_range & y_in_range