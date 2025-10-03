import logging

import dask.array as da
import numpy as np
from exaspim_flatfield_correction.utils.utils import resize

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
    mask: np.ndarray | None = None,
    max_workers: int = 16,
    resize_mode: str = "skimage",
    working_size: int = 256,
    smoothness_flatfield: float = 0.1,
    smoothness_darkfield: float = 0.1,
    max_slices: int = 0,
) -> "BaSiC":
    """Fit a BaSiC flatfield/darkfield model to an image stack.

    Parameters
    ----------
    im : numpy.ndarray
        Image stack with shape ``(frames, y, x)`` or ``(z, y, x)``.
    autotune : bool, default=False
        Whether to run BaSiC's autotuner prior to fitting.
    autotune_iter : int, default=50
        Maximum iterations the autotuner should run.
    get_darkfield : bool, default=False
        If ``True``, estimate the darkfield component in addition to the
        flatfield.
    autosegment : bool, default=False
        Enable BaSiC autosegmentation to focus the fit on bright structures.
    sort_intensity : bool, default=False
        Sort frames by total intensity before fitting.
    shuffle_frames : bool, default=False
        Shuffle frames randomly prior to fitting.
    mask : numpy.ndarray or None, default=None
        Optional weighting mask aligned with ``im``.
    max_workers : int, default=16
        Number of worker threads used by BaSiC.
    resize_mode : str, default="skimage"
        Resize backend used internally by BaSiC.
    working_size : int, default=256
        Working resolution along each spatial axis used by BaSiC.
    smoothness_flatfield : float, default=0.1
        Regularisation strength applied to the flatfield component.
    smoothness_darkfield : float, default=0.1
        Regularisation strength applied to the darkfield component.
    max_slices : int, default=0
        If greater than zero, limit the fit to the ``max_slices`` slices with
        the largest mask area.

    Returns
    -------
    BaSiC
        Fitted BaSiC model instance.
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
    im: da.Array,
    fit: "BaSiC",
    chunks: tuple[int, int] | str = (256, 256),
) -> da.Array:
    """Apply a fitted BaSiC model to a Dask array lazily.

    Parameters
    ----------
    im : dask.array.Array
        Image to correct with shape ``(c, y, x)`` or ``(z, y, x)``.
    fit : BaSiC
        Fitted BaSiC model instance.
    chunks : tuple[int, int] or str, default=(256, 256)
        Chunk specification for the generated flatfield and darkfield arrays.

    Returns
    -------
    dask.array.Array
        Dask array representing the corrected image volume.
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
