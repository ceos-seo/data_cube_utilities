def NDBI(ds):
    """
    Computes the Normalized Difference Built-up Index for an `xarray.Dataset`.
    The formula is (SWIR1 - NIR) / (SWIR1 + NIR)
    Values should be in the range [-1,1] for valid LANDSAT data (swir1 and nir are positive).

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain 'swir1' and 'nir' `DataArrays`.

    Returns
    -------
    ndbi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.
    """
    return (ds.swir1 - ds.nir) / (ds.swir1 + ds.nir)