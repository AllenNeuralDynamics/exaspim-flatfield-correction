import numpy as np
import dask.array as da
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splrep, splev
from exaspim_flatfield_correction.utils.mask_utils import get_mask
import logging

_LOGGER = logging.getLogger(__name__)


def percentile_project(
    arr: "np.ndarray | da.Array", axis: int, percentile: float = 95
) -> np.ndarray:
    """
    Compute a percentile projection along a given axis.

    Parameters
    ----------
    arr : np.ndarray or dask.array.Array
        Input array.
    axis : int
        Axis along which to compute the percentile.
    percentile : float, optional
        Percentile to compute (default is 95).

    Returns
    -------
    np.ndarray
        Percentile projection as a numpy array.
    """
    if isinstance(arr, da.Array):
        arr = arr.compute()
    return np.percentile(arr, percentile, axis=axis)


def fit_splrep(
    x: np.ndarray, y: np.ndarray, smoothing: float = 0, k: int = 3
) -> np.ndarray:
    """
    Fit a smoothing spline to the data using splrep and splev.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Dependent variable.
    smoothing : float, optional
        Smoothing factor; larger values result in a smoother spline.
        Default is 0.
    k : int, optional
        Degree of the spline (default is cubic, k=3).

    Returns
    -------
    np.ndarray
        The fitted spline evaluated at x.
    """
    tck = splrep(x, y, s=smoothing, k=k)
    fitted = splev(x, tck)
    return fitted


def rescale_spline(
    x: np.ndarray,
    y: np.ndarray,
    new_width: int,
    smoothing: float = 0,
    k: int = 3,
) -> np.ndarray:
    """
    Fit a spline to (x, y) and rescale it to a new width.

    Parameters
    ----------
    x : np.ndarray
        Original x values.
    y : np.ndarray
        Original y values.
    new_width : int
        Desired output width.
    smoothing : float, optional
        Smoothing factor for the spline. Default is 0.
    k : int, optional
        Degree of the spline (default is cubic, k=3).

    Returns
    -------
    np.ndarray
        Rescaled spline values at the new width.
    """
    # Fit the spline using the original x and y data
    tck = splrep(x, y, s=smoothing, k=k)

    # Create new x values for the upscaled image
    new_x = np.linspace(0, new_width, new_width)

    # Map new_x into the original domain using a linear transformation
    # Here, original domain is assumed to be from x.min() to x.max()
    # (e.g., 0 to 128)
    scaled_new_x = (new_x - new_x.min()) * (
        (x.max() - x.min()) / new_width
    ) + x.min()

    # Evaluate the spline at the scaled x values
    new_y = splev(scaled_new_x, tck)

    return new_y


def get_correction_func(
    proj: np.ndarray,
    mask_2d: np.ndarray,
    axis: int,
    new_width: int,
    spline_smoothing: float = 0,
    limits: "tuple[float, float] | None" = None,
) -> tuple[np.ndarray, float]:
    """
    Compute a 1D correction profile by fitting a spline to the normalized
    median profile of a projection.

    Parameters
    ----------
    proj : np.ndarray
        2D projection of the image.
    mask_2d : np.ndarray
        2D binary mask.
    axis : int
        Axis along which to compute the profile.
    new_width : int
        Desired output width for the correction profile.
    spline_smoothing : float, optional
        Smoothing factor for the spline. Default is 0.
    limits : tuple of float, optional
        Tuple (min, max) to clip the fitted profile.

    Returns
    -------
    fitted : np.ndarray
        The fitted and rescaled correction profile.
    global_med : float
        The global median value from the projection.
    """
    norm_med, global_med = get_profile(proj, mask_2d, axis)

    # Fit a spline to the normalized median profile using make_splrep.
    fitted = rescale_spline(
        np.arange(len(norm_med)),
        norm_med,
        new_width,
        smoothing=spline_smoothing,
    )
    if limits is not None:
        fitted = np.clip(fitted, limits[0], limits[1])

    return fitted, global_med


def get_profile(
    proj: np.ndarray, mask_2d: np.ndarray, axis: int
) -> tuple[np.ndarray, float]:
    """
    Compute a normalized median profile and global median from a projection
    and mask.

    Parameters
    ----------
    proj : np.ndarray
        2D projection of the image.
    mask_2d : np.ndarray
        2D binary mask.
    axis : int
        Axis along which to compute the profile.

    Returns
    -------
    norm_med : np.ndarray
        Normalized median profile along the specified axis.
    global_med : float
        Global median value from the projection.
    """
    m_nan = np.where(mask_2d, proj, np.nan)

    # Compute a global median from the projection.
    global_med = np.nanmedian(m_nan)
    _LOGGER.info(f"Global median (from 2D projection): {global_med}")
    if global_med == 0:
        return np.ones(shape=(proj.shape[axis])), 0

    # Compute a 1D median profile along the specified axis.
    med = np.nanmedian(m_nan, axis=axis)
    med[np.isnan(med)] = global_med
    norm_med = med / global_med

    return norm_med, global_med


def _debug_correction(
    arr: "np.ndarray | da.Array",
    threshold: float = 0,
    corr_axis: str = "x",
    spline_smoothing: float = 0,
    mask: "np.ndarray | None" = None,
    proj_mask: "np.ndarray | None" = None,
    limits: "tuple[float, float] | None" = None,
    p: float = 95,
    factor: float = 1,
    sigma: float = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Correct a 3D image by estimating and removing an intensity gradient along
    the chosen axis using a spline fit.

    Parameters
    ----------
    arr : np.ndarray or dask.array.Array, shape (Z, Y, X)
        The 3D image data.
    threshold : float, optional
        Threshold for creating a mask. Default is 0.
    corr_axis : {'x', 'y', 'z'}, optional
        The axis along which to perform the correction. Default is 'x'.
    spline_smoothing : float, optional
        Smoothing factor for the spline fit. Default is 0.
    mask : np.ndarray, optional
        A 3D binary mask. If None, one will be created using the threshold.
    proj_mask : np.ndarray, optional
        Optional 2D mask for the projection. If None, one will be created.
    limits : tuple of float, optional
        Tuple (min, max) to clip the fitted profile.
    p : float, optional
        Percentile for projection. Default is 95.
    factor : float, optional
        Clipping factor for the input image. Default is 1.
    sigma : float, optional
        Standard deviation for Gaussian blur. Default is 1.

    Returns
    -------
    corrected_im : np.ndarray
        The corrected 3D image. Pixels below threshold remain unchanged.
    fitted : np.ndarray
        The fitted 1D correction profile.
    norm_med : np.ndarray
        The normalized median profile used for fitting.
    m : np.ndarray
        The 2D projection used in the estimation.
    global_med : float
        The global median computed from the 2D projection.
    mask : np.ndarray
        The 3D binary mask where arr > threshold.
    """
    if isinstance(arr, da.Array):
        arr = arr.compute().astype(np.float32)

    # Define projection and median axes and reshaping for the correction
    # profile.
    if corr_axis.lower() == "x":
        proj_axis = 0  # Collapse Z: m becomes (Y, X)
        median_axis = 0  # Median over Y gives profile of length X
        reshape_correction = lambda prof: prof.reshape(1, 1, -1)
    elif corr_axis.lower() == "z":
        proj_axis = 2  # Collapse X: m becomes (Z, Y)
        median_axis = 1  # Median over X gives profile of length Z
        reshape_correction = lambda prof: prof.reshape(-1, 1, 1)
    elif corr_axis.lower() == "y":
        proj_axis = 2  # Collapse X: m becomes (Z, Y)
        median_axis = 0  # Median over Z gives profile of length Y
        reshape_correction = lambda prof: prof.reshape(1, -1, 1)
    else:
        raise ValueError("corr_axis must be either 'x', 'y', or 'z'")

    arr_med = np.nanmedian(np.where(mask, arr, np.nan))
    arr_clipped = np.clip(arr, 0, arr_med * factor)

    # Compute a 2D projection of the 3D array.
    m = percentile_project(arr_clipped, proj_axis, p)

    # Apply a Gaussian blur to smooth the projection.
    m = gaussian_filter(m, sigma=sigma)

    # Create a 2D mask based on the threshold from the projection and replace
    # values below threshold with NaN.
    if proj_mask is None:
        mask_2d = get_mask(m, threshold, min_size=1000)
    else:
        mask_2d = proj_mask
    m_nan = np.where(mask_2d, m, np.nan)

    # Compute a global median from the projection.
    global_med = np.nanmedian(m_nan)
    print("Global median (from 2D projection):", global_med)

    # Compute a 1D median profile along the specified axis.
    med = np.nanmedian(m_nan, axis=median_axis)
    med[np.isnan(med)] = global_med
    norm_med = med / global_med

    # Fit a spline to the normalized median profile using make_splrep.
    fitted = fit_splrep(
        np.arange(len(norm_med)), norm_med, smoothing=spline_smoothing
    )
    if limits is not None:
        fitted = np.clip(fitted, limits[0], limits[1])

    # Reshape the fitted profile to build the correction array.
    correction = reshape_correction(fitted)

    # Create a 3D mask for the input image if one was not provided.
    if mask is None:
        mask = get_mask(arr, threshold, min_size=None)

    # Apply the correction only where the mask is True.
    corrected_im = np.where(mask, arr / correction, arr)

    return corrected_im, fitted, norm_med, m, global_med, mask
