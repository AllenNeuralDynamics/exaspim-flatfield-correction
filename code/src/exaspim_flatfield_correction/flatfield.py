import os
import logging
from typing import List

import numpy as np
import dask.array as da
import dask
from dask_image.ndfilters import gaussian_filter

from exaspim_flatfield_correction.utils.zarr_utils import store_ome_zarr
from exaspim_flatfield_correction.utils.utils import (
    resize,
    resize_dask,
    get_abs_path,
)

_LOGGER = logging.getLogger(__name__)

try:
    from basicpy import BaSiC
except ModuleNotFoundError:
    _LOGGER.warning(
        "BaSiC not installed. Please install it with 'pip install basicpy' to "
        "use the BaSiC flatfield correction."
    )


def fit_basic(
    im: np.ndarray,
    autotune: bool = False,
    get_darkfield: bool = False,
    autosegment: bool = False,
    sort_intensity: bool = False,
    shuffle_frames: bool = False,
    mask: np.ndarray = None,
    max_workers: int = 16,
    resize_mode: str = "skimage",
    working_size: int = 128,
    smoothness_flatfield: float = 0.1,
    smoothness_darkfield: float = 0.1,
) -> "BaSiC":
    """
    Fit a BaSiC flatfield/darkfield correction model to an image stack.

    Parameters
    ----------
    im : np.ndarray
        Image stack (frames, Y, X) or (Z, Y, X).
    autotune : bool, optional
        Whether to autotune BaSiC parameters. Default is False.
    get_darkfield : bool, optional
        Whether to estimate darkfield. Default is False.
    autosegment : bool, optional
        Whether to use autosegmentation. Default is False.
    sort_intensity : bool, optional
        Whether to sort frames by intensity before fitting. Default is False.
    shuffle_frames : bool, optional
        Whether to shuffle frames before fitting. Default is False.
    mask : np.ndarray, optional
        Optional mask for fitting weighting.
    max_workers : int, optional
        Number of workers for parallel processing. Default is 16.
    resize_mode : str, optional
        Resize mode for BaSiC. Default is 'skimage'.
    working_size : int, optional
        Working size for BaSiC. Default is 128.
    smoothness_flatfield : float, optional
        Smoothness parameter for flatfield. Default is 0.1.
    smoothness_darkfield : float, optional
        Smoothness parameter for darkfield. Default is 0.1.

    Returns
    -------
    BaSiC
        Fitted BaSiC model object.
    """
    basic = BaSiC(
        autosegment=autosegment,
        sort_intensity=sort_intensity,
        get_darkfield=get_darkfield,
        max_workers=max_workers,
        resize_mode=resize_mode,
        working_size=working_size,
        smoothness_flatfield=smoothness_flatfield,
        smoothness_darkfield=smoothness_darkfield,
    )
    if shuffle_frames:
        im = im.copy()
        np.random.shuffle(im)
    if autotune:
        basic.autotune(im, early_stop=True, n_iter=50)
        _LOGGER.info(
            f"Autotune: flatfield={basic.smoothness_flatfield}, \
            darkfield={basic.smoothness_darkfield}, \
            sparse_cost={basic.sparse_cost_darkfield}"
        )

    basic.fit(im, fitting_weight=mask)

    return basic


def transform_basic(im: da.Array, fit: "BaSiC", chunks: tuple = (256, 256)) -> da.Array:
    """
    Apply a fitted BaSiC flatfield/darkfield correction to an image.

    Parameters
    ----------
    im : dask.array.Array
        Image to correct (C, Y, X) or (Z, Y, X).
    fit : BaSiC
        Fitted BaSiC model object.
    chunks : tuple, optional
        Chunk size for Dask arrays. Default is (256, 256).

    Returns
    -------
    dask.array.Array
        Flatfield/darkfield corrected image.
    """
    flatfield = da.from_array(
        resize(fit.flatfield, im.shape[-2:]), chunks=chunks
    )
    darkfield = da.from_array(
        resize(fit.darkfield, im.shape[-2:]), chunks=chunks
    )

    return (im.astype(np.float32) - darkfield[np.newaxis]) / flatfield[
        np.newaxis
    ]


def sigmoid_correction(
    tile_da: da.Array,
    overlap_percent: float = 15,
    k: float = 0.5,
    baseline: float = 0.75,
) -> da.Array:
    """
    Apply a sigmoid-based correction to a tile to compensate for edge effects.

    Parameters
    ----------
    tile_da : dask.array.Array
        Input tile image (C, Y, X) or (Z, Y, X).
    overlap_percent : float, optional
        Percent overlap for the sigmoid transition. Default is 15.
    k : float, optional
        Steepness of the sigmoid. Default is 0.5.
    baseline : float, optional
        Baseline value for normalization. Default is 0.75.

    Returns
    -------
    dask.array.Array
        Corrected image after sigmoid adjustment.
    """
    xo = int(tile_da.shape[-1] * (1 - overlap_percent / 100))
    xprofile = np.linspace(
        int(-tile_da.shape[2] / 2), int(tile_da.shape[2] / 2), tile_da.shape[2]
    )  # lateral lineprofile of image
    correction = (
        1 / (1 + np.exp(-k * (xprofile - (-tile_da.shape[2] / 2 + xo))))
        + baseline
    )
    correction = correction / np.min(correction)  # normalize so baseline is 1
    # copy sigmoid into 2D matrix
    flatfield = da.from_array(
        np.tile(correction, (tile_da.shape[1], 1)).astype("float32"),
        chunks=tile_da.chunksize[-2:],
    )
    # do flatfield correction
    im = tile_da * flatfield[np.newaxis]
    return da.clip(im, 0, 2**16 - 1)


def gaussian_flatfield(im: da.Array, sigma: float) -> da.Array:
    """
    Estimate shading using a Gaussian filter for flatfield correction.

    Parameters
    ----------
    im : dask.array.Array
        Input image (normalized to [0, 1]).
    sigma : float
        Standard deviation for Gaussian filter.

    Returns
    -------
    dask.array.Array
        Shading image estimated by Gaussian filtering.
    """
    im = normalize_uint16(im)
    # Gaussian filtering for shading
    shading = gaussian_filter(im, sigma, mode="reflect")
    return shading


def gaussian_correction(
    out_res: int,
    gauss_res: int,
    sigma: float,
    subtracted_low_res: da.Array,
    subtracted_full_res: da.Array,
    mean_value: float,
    tmpdir: str = "../results/tmp",
) -> da.Array:
    """
    Apply Gaussian-based flatfield correction to an image using
    a low-res shading estimate.

    Parameters
    ----------
    out_res : int
        Output resolution level.
    gauss_res : int
        Resolution level for Gaussian shading.
    sigma : float
        Standard deviation for Gaussian filter.
    subtracted_low_res : dask.array.Array
        Low-resolution background-subtracted image.
    subtracted_full_res : dask.array.Array
        Full-resolution background-subtracted image.
    mean_value : float
        Mean value for normalization.
    tmpdir : str, optional
        Temporary directory for storing intermediate results.
        Default is '../results/tmp'.

    Returns
    -------
    dask.array.Array
        Flatfield-corrected image.
    """
    shading = gaussian_flatfield(subtracted_low_res, sigma)
    shading = shading.rechunk(64, 64, 64)

    tmpdir_abs = get_abs_path(tmpdir)
    store_ome_zarr(
        shading, os.path.join(tmpdir_abs, "shading.zarr"), overwrite=True
    )
    shading = da.from_zarr(
        os.path.join(tmpdir_abs, "shading.zarr/0")
    ).squeeze()

    min_value = shading.min().compute()
    scale_factor = 2 ** (gauss_res - out_res)
    shading_upscaled = resize_dask(shading, scale_factor=scale_factor)
    # avoid division by zero
    shading_upscaled = da.maximum(shading_upscaled, min_value)

    full_res = normalize_uint16(subtracted_full_res)
    corrected = full_res * mean_value / shading_upscaled

    return corrected * 65535


def subtract_bkg(im: da.Array, bkg_da: da.Array) -> da.Array:
    """
    Subtract a background image from an image and clip to valid range.

    Parameters
    ----------
    im : dask.array.Array
        Input image.
    bkg_da : dask.array.Array
        Background image to subtract.

    Returns
    -------
    dask.array.Array
        Background-subtracted and clipped image.
    """
    return da.clip(im - bkg_da, 0, 2**16 - 1)


def rescale_dask(
    im_dask: da.Array, new_min: int = 0, new_max: int = 65535
) -> da.Array:
    """
    Rescale a Dask array to a new min and max value using min-max scaling.

    Parameters
    ----------
    im_dask : dask.array.Array
        Input image as a Dask array.
    new_min : int, optional
        New minimum value. Default is 0.
    new_max : int, optional
        New maximum value. Default is 65535.

    Returns
    -------
    dask.array.Array
        Rescaled image as a Dask array (uint16).
    """
    # Compute the actual min and max of the dask array
    min_value, max_value = dask.compute(im_dask.min(), im_dask.max())
    # Apply min-max scaling
    scaled_image = (im_dask - min_value) / (max_value - min_value) * (
        new_max - new_min
    ) + new_min
    return scaled_image.astype("uint16")


def normalize_uint16(im_dask_uint16: da.Array) -> da.Array:
    """
    Normalize a uint16 Dask array to the range [0, 1] as float32.

    Parameters
    ----------
    im_dask_uint16 : dask.array.Array
        Input uint16 image as a Dask array.

    Returns
    -------
    dask.array.Array
        Normalized image as float32 in [0, 1].
    """
    normalized_image = im_dask_uint16.astype(np.float32) / 65535.0
    return normalized_image


def min_max_scale(
    im: np.ndarray, new_min: float = 0.01, new_max: float = 1.0
) -> np.ndarray:
    """
    Min-max scale a numpy array to a new range.

    Parameters
    ----------
    im : np.ndarray
        Input image array.
    new_min : float, optional
        New minimum value. Default is 0.01.
    new_max : float, optional
        New maximum value. Default is 1.0.

    Returns
    -------
    np.ndarray
        Scaled image array.
    """
    im = im.astype(np.float32)

    max_val = im.max()
    min_val = im.min()

    return new_min + (new_max - new_min) * (im - min_val) / (max_val - min_val)


def array_mean(arrays: List[da.Array]) -> float:
    """
    Stack a list of 3D Dask arrays and calculate the mean value of
    the stacked array.

    Parameters
    ----------
    arrays : List[dask.array.Array]
        A list of 3D Dask arrays. All arrays should have the same shape and
        data type (uint16).

    Returns
    -------
    float
        The mean value of the stacked Dask array.
    """
    stacked_array = da.stack(arrays, axis=0)

    mean_value = stacked_array.mean().compute()

    return mean_value


def estimate_decay(
    im: da.Array, num_slices: int = 100, threshold: float = 70
) -> float:
    """
    Estimate the decay factor in a 3D image stack by comparing mean intensity
    at the start and end.

    Parameters
    ----------
    im : dask.array.Array
        Input image stack.
    num_slices : int, optional
        Number of slices to use at each end. Default is 100.
    threshold : float, optional
        Intensity threshold for masking. Default is 70.

    Returns
    -------
    float
        Estimated decay factor (fractional reduction).
    """
    masked_image = da.where(im < threshold, im, np.nan)

    start_mean_intensity = da.nanmean(
        masked_image[:num_slices, :, :], axis=(0, 1, 2)
    )

    end_mean_intensity = da.nanmean(masked_image[-num_slices:], axis=(0, 1, 2))

    reduction = (
        start_mean_intensity - end_mean_intensity
    ) / start_mean_intensity

    return reduction.compute()


def z_correction(
    im: da.Array,
    decay_factor: float = "auto",
    num_slices: int = 100,
    threshold: float = 70,
) -> da.Array:
    """
    Apply Z-correction to a 3D image stack based on estimated decay.

    Parameters
    ----------
    im : dask.array.Array
        Input image stack.
    decay_factor : float or 'auto', optional
        Decay factor to use, or 'auto' to estimate. Default is 'auto'.
    num_slices : int, optional
        Number of slices to use for decay estimation. Default is 100.
    threshold : float, optional
        Intensity threshold for masking. Default is 70.

    Returns
    -------
    dask.array.Array
        Z-corrected image stack.
    """
    if decay_factor == "auto":
        decay_factor = estimate_decay(im, num_slices, threshold)
    _LOGGER.info(f"Decay factor: {decay_factor}")
    if np.isnan(decay_factor):
        raise ValueError(
            "decay factor is nan. You probably need a higher threshold"
        )

    mu = decay_factor / im.shape[0]

    # Generate an array of z-coordinates
    z_coords = np.arange(im.shape[0])

    # Apply the function to the z-coordinates
    z_values = np.exp(z_coords * mu)

    # Reshape z_values for broadcasting
    z_values_reshaped = z_values.reshape(im.shape[0], 1, 1)

    # Apply the transformation to the image
    result_image = im.astype(np.float32) * z_values_reshaped

    return da.clip(result_image, 0, 2**16 - 1)
