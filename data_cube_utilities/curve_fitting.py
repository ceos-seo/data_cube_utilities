import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import spline, CubicSpline
from scipy.ndimage.filters import gaussian_filter1d

def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussian_fit(x, y, x_smooth=None, n_pts=200):
    """
    Fits a Gaussian to some data - x and y. Returns predicted interpolation values.
    
    Parameters
    ----------
    x: list-like
        The x values of the data to fit to.
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    """
    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), n_pts)
    mean, sigma = np.nanmean(y), np.nanstd(y)
    popt,pcov = curve_fit(gauss,x,y,p0=[1,mean,sigma], maxfev=np.iinfo(np.int32).max)
    return gauss(x_smooth,*popt)

def gaussian_filter_fit(x, y, x_smooth=None, n_pts=200, sigma=None):
    """
    Fits a Gaussian filter to some data - x and y. Returns predicted interpolation values.
    Currently, smoothing is achieved by fitting a cubic spline to the gaussian filter fit
    of `x` and `y`.

    Parameters
    ----------
    x: list-like
        The x values of the data to fit to.
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like, optional
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int, optional
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    sigma: numeric, optional
        The standard deviation of the Gaussian kernel. A larger value yields a smoother curve,
        but also reduced the closeness of the fit. By default, it is `4 * np.std(y)`.
    """
    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), n_pts)
    sigma = sigma if sigma is not None else 4 * np.std(y)
    gauss_filter_y = gaussian_filter1d(y, sigma)
    cs = CubicSpline(x, gauss_filter_y)
    y_smooth = cs(x_smooth)
    return y_smooth

def poly_fit(x, y, degree, x_smooth=None, n_pts=200):
    """
    Fits a polynomial of any positive integer degree to some data - x and y. Returns predicted interpolation values.
    
    Parameters
    ----------
    x: list-like
        The x values of the data to fit to.
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    degree: int
        The degree of the polynomial to fit.
    """
    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), n_pts)
    return np.array([np.array([coef*(x_val**current_degree) for coef, current_degree in 
                               zip(np.polyfit(x, y, degree), range(degree, -1, -1))]).sum() for x_val in x_smooth])