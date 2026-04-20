import logging
from typing import Any

import dask.array as da
import numpy as np

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
        across z) and the second element is the indices of slices retained for
        background modeling.
    """

    # -- Sanity checks --
    if im.ndim != 3:
        raise ValueError("Input 'im' must be a 3D array with shape (z, y, x).")

    # Convert to float32 if not already
    im = im.astype(np.float32)

    std_z = np.std(im, axis=(1, 2))
    slice_mask = std_z <= np.percentile(std_z, 5)

    retained_indices = np.flatnonzero(slice_mask)
    im = im[slice_mask, :, :]  # shape now could be (z', y, x)
    initial = im.copy()
    initial_indices = retained_indices.copy()

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
        retained_indices = retained_indices[keep_mask]
        im = im[keep_mask, :, :]

        if np.mean(inds_remove) < 0.0001:
            break

    _LOGGER.info(f"N Slices in final stack: {im.shape[0]}")
    if im.shape[0] == 0:
        _LOGGER.warning(
            "Could not estimate background from tile, using initial guess."
        )
        im = initial
        retained_indices = initial_indices

    mu_final = np.median(im, axis=0).astype(np.float32)  # (y, x)

    return mu_final, retained_indices.astype(np.int64, copy=False)


def _z_scale_and_translation(
    transformations: dict[str, Any] | None,
) -> tuple[float, float] | None:
    """Extract the Z scale/translation from OME-Zarr transformations."""

    if not transformations:
        return None

    scale = transformations.get("scale")
    if scale is None or len(scale) < 3:
        return None

    translation = transformations.get("translation")
    z_translation = 0.0
    if translation is not None and len(translation) >= 3:
        z_translation = float(translation[-3])

    z_scale = float(scale[-3])
    if not np.isfinite(z_scale) or z_scale <= 0:
        return None
    if not np.isfinite(z_translation):
        return None

    return z_scale, z_translation


def map_slice_indices_to_target(
    source_indices: np.ndarray,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    source_transformations: dict[str, Any] | None = None,
    target_transformations: dict[str, Any] | None = None,
) -> np.ndarray:
    """Map source Z slices to all target Z planes covered by those slices.

    The returned indices are in the target array coordinate system. When both
    source and target OME-Zarr scale metadata are present, mapping is performed
    in physical/world coordinates. Otherwise, it falls back to the ratio of
    target and source Z sizes.
    """

    source_indices = np.asarray(source_indices, dtype=np.int64)
    if source_indices.ndim != 1:
        raise ValueError("source_indices must be a 1D array.")
    if len(source_shape) != 3 or len(target_shape) != 3:
        raise ValueError(
            "source_shape and target_shape must be 3D ZYX shapes."
        )

    source_z = int(source_shape[0])
    target_z = int(target_shape[0])
    if source_z <= 0 or target_z <= 0:
        raise ValueError("source and target Z dimensions must be positive.")
    if np.any(source_indices < 0) or np.any(source_indices >= source_z):
        raise ValueError("source_indices contain slices outside source_shape.")

    source_transform = _z_scale_and_translation(source_transformations)
    target_transform = _z_scale_and_translation(target_transformations)
    use_metadata = (
        source_transform is not None and target_transform is not None
    )

    if use_metadata:
        source_scale, source_translation = source_transform
        target_scale, target_translation = target_transform
    else:
        raise ValueError(
            "OME-Zarr scale/translation metadata unavailable for background "
            "slice mapping; falling back to shape-ratio mapping."
        )

    mapped_indices: list[int] = []
    eps = 1e-9
    for source_index in source_indices:
        source_start = source_translation + float(source_index) * source_scale
        source_stop = (
            source_translation + float(source_index + 1) * source_scale
        )

        start_index = int(
            np.floor((source_start - target_translation) / target_scale + eps)
        )
        stop_index = int(
            np.ceil((source_stop - target_translation) / target_scale - eps)
        )

        start_index = max(0, start_index)
        stop_index = min(target_z, stop_index)
        if stop_index > start_index:
            mapped_indices.extend(range(start_index, stop_index))

    return np.unique(np.asarray(mapped_indices, dtype=np.int64))


def _rechunk_for_plane_median(
    im: da.Array,
    max_spatial_chunk: int,
) -> da.Array:
    """Use one Z chunk and capped spatial chunks for a plane-wise median."""

    spatial_chunks = []
    for axis in (1, 2):
        existing = (
            int(im.chunks[axis][0]) if im.chunks else int(im.shape[axis])
        )
        spatial_chunks.append(
            max(1, min(int(im.shape[axis]), existing, max_spatial_chunk))
        )

    return im.rechunk((int(im.shape[0]), *spatial_chunks))


def estimate_bkg_from_mapped_slices(
    target_im: da.Array,
    source_slice_indices: np.ndarray,
    source_shape: tuple[int, ...],
    source_transformations: dict[str, Any] | None = None,
    target_transformations: dict[str, Any] | None = None,
    max_spatial_chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a target-resolution background from mapped source slices.

    Parameters
    ----------
    target_im : dask.array.Array, shape (z, y, x)
        Target-resolution image volume from which the final background image
        should be constructed.
    source_slice_indices : np.ndarray
        Background-dominant source-resolution Z slice indices.
    source_shape : tuple[int, ...]
        Shape of the source-resolution volume used to select the slices.
    source_transformations : dict or None, default=None
        OME-Zarr scale/translation metadata for the selector volume.
    target_transformations : dict or None, default=None
        OME-Zarr scale/translation metadata for ``target_im``.
    max_spatial_chunk : int, default=512
        Maximum Y/X chunk edge length to use when computing the Dask median.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The target-resolution 2D median background and the mapped target Z
        indices used to compute it.
    """

    if target_im.ndim != 3:
        raise ValueError("target_im must be a 3D array with shape (z, y, x).")
    if max_spatial_chunk <= 0:
        raise ValueError("max_spatial_chunk must be positive.")

    target_indices = map_slice_indices_to_target(
        source_slice_indices,
        source_shape,
        target_im.shape,
        source_transformations=source_transformations,
        target_transformations=target_transformations,
    )
    if target_indices.size == 0:
        raise ValueError(
            "Background source slices did not map to any target Z planes."
        )

    _LOGGER.info(
        "Mapped %s source background slices to %s target planes.",
        len(source_slice_indices),
        len(target_indices),
    )

    selected = da.take(target_im.astype(np.float32), target_indices, axis=0)
    selected = _rechunk_for_plane_median(selected, max_spatial_chunk)
    bkg = da.median(selected, axis=0).astype(np.float32).compute()

    return bkg.astype(np.float32, copy=False), target_indices


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
