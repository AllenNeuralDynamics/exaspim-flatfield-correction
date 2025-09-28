import logging

import numpy as np
import dask.array as da

from exaspim_flatfield_correction.utils.utils import (
    resize
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
        _LOGGER.info(
            "Shape after filtering by mask area: "
            f"{im.shape}, mask shape: {mask.shape}"
        )

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


def transform_basic(
    im: da.Array, fit: "BaSiC", chunks: tuple = (256, 256)
) -> da.Array:
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
