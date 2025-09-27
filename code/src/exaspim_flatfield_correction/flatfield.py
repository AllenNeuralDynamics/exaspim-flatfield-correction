import os
import logging

import numpy as np
import dask.array as da
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
    autotune_iter: int = 50,
    get_darkfield: bool = False,
    autosegment: bool = False,
    sort_intensity: bool = False,
    shuffle_frames: bool = False,
    mask: np.ndarray = None,
    max_workers: int = 16,
    resize_mode: str = "skimage",
    working_size: int = 256,
    smoothness_flatfield: float = 0.1,
    smoothness_darkfield: float = 0.1,
    max_slices: int = 0,
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
        Working size for BaSiC. Default is 256.
    smoothness_flatfield : float, optional
        Smoothness parameter for flatfield. Default is 0.1.
    smoothness_darkfield : float, optional
        Smoothness parameter for darkfield. Default is 0.1.
    max_slices : int, optional
        If > 0, use only the top N mask slices (by area) and corresponding image slices. Default is 0 (use all).

    Returns
    -------
    BaSiC
        Fitted BaSiC model object.
    """
    # If max_slices > 0 and mask is provided, select top N slices by mask area
    if max_slices > 0 and mask is not None:
        # Compute area for each mask slice (sum over each 2D mask)
        mask_areas = mask.sum(axis=(1, 2))
        # Get indices of top N slices by area (descending)
        top_indices = np.argsort(mask_areas)[::-1][:max_slices]
        # Sort indices to preserve order
        top_indices = np.sort(top_indices)
        im = im[top_indices]
        mask = mask[top_indices]
        _LOGGER.info("Shape after filtering by mask area: "
                     f"{im.shape}, mask shape: {mask.shape}")

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
        basic.autotune(im, early_stop=True, n_iter=autotune_iter)
        _LOGGER.info(
            f"Autotune: flatfield={basic.smoothness_flatfield}, "
            f"darkfield={basic.smoothness_darkfield}, "
            f"sparse_cost={basic.sparse_cost_darkfield}"
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
