import xarray as xr
import numpy as np

def generate_gradient(matrix,resolution = 1.0, remove_border = False):
	north, east = np.gradient(matrix)
	max_gradient = np.maximum.reduce([abs(north), abs(east)])
	if remove_border:
		max_gradient[:,0] = np.nan
		max_gradient[:,-1] = np.nan
		max_gradient[0, :] = np.nan
		max_gradient[-1,:] = np.nan
	return (max_gradient/float(resolution))

def generate_degree(dem_matrix, resolutution = 1.0):
	return np.rad2deg(np.arctan(generate_gradient(dem_matrix, resolution = resolution)))

def create_slope_mask(dem_data, resolution = 1.0, degree_threshold = 15, no_data = -9999):
	## Uses values at first DEM acquistion date 
	target = dem_data.dem.values[0].astype(np.float32)
	target[target == no_data] = np.nan
	## Generates gradient per dem pixel, turns to degrees per dem pixel, bounds to range between 1 and 100
	slopes = generate_gradient(target, resolution = resolution)
	angle_of_elevation = np.rad2deg(np.arctan(slopes))
	## Create a mask for greater than 15 degrees. Here is what 15 degrees looks like: https://i.stack.imgur.com/BIrAW.png
	mask = angle_of_elevation > degree_threshold
	return mask
