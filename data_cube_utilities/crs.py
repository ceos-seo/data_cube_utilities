# from pyproj import Transformer
# from functools import partial

# def xarray_change_crs(ds, out_crs, in_crs=None, x_dim=None, y_dim=None):
#     """
#     (In development)
    
#     Changes the CRS of the x/y coordinates of an `xarray.Dataset` or `xarray.DataArray`.
#     This is an in-place operation.
    
#     Parameters
#     ----------
#     ds: xarray.Dataset or xarray.DataArray
#         The xarray object to change coordinate values for.
#     out_crs, in_crs: str
#         The output and input CRSes. If `in_crs` is not specified, the function will 
#         attempt to read the metadata of `ds` to determine it. If this fails, the function 
#         will throw an error.
#     x_dim, y_dim: str
#         The string names of the x and y dimensions in `ds`. If not specified, the dimension
#         name will be inferred. If this fails, the function will throw an error.
        
#     Returns
#     -------
#     out_ds: xarray.Dataset or xarray.DataArray
#         Same as `ds` but with different coordinate values.
#     """
#     if in_crs is None:
#         in_crs = meas_data.attrs.get('crs')
#         assert in_crs is not None, 'Could not determine `in_crs`. Please specify this argument.'
#         in_crs = meas_data.crs[6:] # Remove the leading '+init='.
#     if x_dim is None:
#         x_dim = 'x' if 'x' in ds.dims else None
#         x_dim = 'longitude' if x_dim is None and 'longitude' in ds.dims else x_dim
#         assert x_dim is not None, 'Could not determine `x_dim`. Please specify this argument.'
#     if y_dim is None:
#         y_dim = 'y' if 'y' in ds.dims else None
#         y_dim = 'latitude' if y_dim is None and 'latitude' in ds.dims else y_dim
#         assert y_dim is not None, 'Could not determine `y_dim`. Please specify this argument.'
#     x_coords = ds.coords[x_dim]
#     y_coords = ds.coords[y_dim]
#     transformer = Transformer.from_crs(in_crs, out_crs)
#     new_x_coords, new_y_coords = [], []
#     for ind, (x_val, y_val) in enumerate(zip(x_coords.values, y_coords.values)):
#         x_val, y_val = transformer.transform(x_val, y_val)
#         new_x_coords.append(x_val)
#         new_y_coords.append(y_val)
#     ds.assign_coords({x_dim:new_x_coords})
#     ds.assign_coords({y_dim:new_y_coords})