import numpy as np
import logging

import dask.array as da

_LOGGER = logging.getLogger(__name__)


def estimate_bkg(
    im: np.ndarray,
    sigma_factor: float = 3.0,
    prob_thresh: float = 0.01,
    n_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat-field background estimation for scattered light with
    variable background, processing a 3D array.

    Parameters
    ----------
    im : np.ndarray, shape (z, y, x)
        3D array where each slice is im[z, :, :].
        Typically loaded from Zarr or elsewhere.
    sigma_factor : float
        Multiplier for pixel-level std to identify high pixels.
    prob_thresh : float
        Probability threshold: fraction of pixels in a slice that
        exceed (mu + sigma_factor*sigma).
    n_iter : int
        Number of iterations for outlier slice removal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple where the first element is the estimated 2D background (median
        across z) and the second element is the filtered stack of background
        slices used for subsequent modelling.
    """

    # -- Sanity checks --
    if im.ndim != 3:
        raise ValueError("Input 'im' must be a 3D array with shape (z, y, x).")

    # Convert to float32 if not already
    im = im.astype(np.float32)

    std_z = np.std(im, axis=(1, 2))
    slice_mask = std_z <= np.percentile(std_z, 5)

    im = im[slice_mask, :, :]  # shape now could be (z', y, x)
    initial = im.copy()

    # Iterative outlier-slice removal
    for _ in range(n_iter):
        # Compute mean & std across slices (axis=0) => shape (y, x)
        # Remember, after removing slices, we have new z' dimension
        mu = np.median(im)
        sigma = np.std(im)

        # fraction of "high" pixels in each slice => shape (z',)
        threshold_2d = mu + sigma_factor * sigma  # shape (y, x)
        # Broadcast threshold over each slice
        high_count = im > threshold_2d  # shape (z', y, x)
        frac_high = np.mean(high_count, axis=(1, 2))

        # We remove slices if:
        #   fraction of high pixels > prob_thresh
        inds_remove = frac_high > prob_thresh

        keep_mask = ~inds_remove
        im = im[keep_mask, :, :]

        if np.mean(inds_remove) < 0.0001:
            break

    _LOGGER.info(f"N Slices in final stack: {im.shape[0]}")
    if im.shape[0] == 0:
        _LOGGER.warning(
            "Could not estimate background from tile, using initial guess."
        )
        im = initial

    mu_final = np.median(im, axis=0).astype(np.float32)  # (y, x)

    return mu_final, im


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
    return da.clip(im - bkg_da, 0, None)
