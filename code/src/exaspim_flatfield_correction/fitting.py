import logging
from collections.abc import Iterable

import dask.array as da
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.ndimage import gaussian_filter1d

_LOGGER = logging.getLogger(__name__)


def rescale_spline(
    x: np.ndarray,
    y: np.ndarray,
    new_width: int,
    smoothing: float = 0,
    k: int = 3,
) -> np.ndarray:
    """Fit a smoothing spline and evaluate it on a new uniform grid.

    Parameters
    ----------
    x : numpy.ndarray
        Sample locations for the original profile.
    y : numpy.ndarray
        Profile values evaluated at ``x``.
    new_width : int
        Number of samples desired in the rescaled profile.
    smoothing : float, default=0
        Smoothing factor passed to ``scipy.interpolate.splrep``.
    k : int, default=3
        Spline degree used when fitting.

    Returns
    -------
    numpy.ndarray
        Resampled profile of length ``new_width``.
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


def generate_axis_fit(
    profile: "np.ndarray | da.Array",
    new_width: int,
    smoothing: float,
    limits: "tuple[float, float] | None" = None,
) -> np.ndarray:
    """Resample and optionally clip a 1D correction profile.

    Parameters
    ----------
    profile : numpy.ndarray or dask.array.Array
        Normalised 1D profile to be rescaled.
    new_width : int
        Desired number of samples after rescaling.
    smoothing : float
        Smoothing factor forwarded to :func:`rescale_spline`.
    limits : tuple of float or None, default=None
        When provided, clamp the resampled profile to ``(lower, upper)``.

    Returns
    -------
    numpy.ndarray
        Resampled and optionally clipped profile.
    """
    if isinstance(profile, da.Array):
        profile_np = profile.astype(np.float32).compute()
    else:
        profile_np = np.asarray(profile, dtype=np.float32)
    x = np.arange(profile_np.size, dtype=np.float32)
    fitted = rescale_spline(x, profile_np, new_width, smoothing=smoothing)
    if limits is not None:
        fitted = np.clip(fitted, limits[0], limits[1])
    return fitted


def compute_axis_fits(
    volume: "np.ndarray | da.Array",
    mask: "np.ndarray | da.Array",
    full_shape: tuple[int, int, int],
    *,
    smooth_sigma: "float | None" = None,
    percentile: "float | None" = None,
    min_voxels: int = 0,
    spline_smoothing: float = 0,
    limits_x: "tuple[float, float] | None" = None,
    limits_y: "tuple[float, float] | None" = None,
    limits_z: "tuple[float, float] | None" = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute axis-aligned correction profiles and corresponding fits.

    Parameters
    ----------
    volume : numpy.ndarray or dask.array.Array
        Low-resolution image volume used to estimate the correction.
    mask : numpy.ndarray or dask.array.Array
        Foreground mask aligned with ``volume``.
    full_shape : tuple of int
        Target shape of the full-resolution dataset (z, y, x).
    smooth_sigma : float or None, default=None
        Optional Gaussian sigma for smoothing the masked profiles prior to
        normalisation.
    percentile : float or None, default=None
        Percentile used to summarise intensities along each axis; defaults to
        the median.
    min_voxels : int, default=0
        Minimum number of mask voxels required in a slice to utilise its
        percentile measurement.
    spline_smoothing : float, default=0
        Smoothing factor applied when resampling the correction profiles.
    limits_x, limits_y, limits_z : tuple of float or None, default=None
        Optional clamp bounds applied to the resampled profiles along each
        axis.

    Returns
    -------
    tuple of (dict[str, numpy.ndarray], dict[str, float])
        Mapping of axis labels to fitted correction curves and a mapping of
        axis labels to the corresponding global medians.
    """

    axis_specs = (
        ("x", 2, full_shape[2], limits_x),
        ("y", 1, full_shape[1], limits_y),
        ("z", 0, full_shape[0], limits_z),
    )

    axis_indices = [axis_idx for _, axis_idx, _, _ in axis_specs]
    profiles, global_med = masked_axis_profile(
        volume,
        mask,
        axes=axis_indices,
        smooth_sigma=smooth_sigma,
        percentile=percentile,
        min_voxels=min_voxels,
    )

    fits: dict[str, np.ndarray] = {}
    medians: dict[str, float] = {}

    for axis_label, axis_idx, width, limits in axis_specs:
        profile = profiles.get(axis_idx)
        if profile is None:
            raise ValueError(f"Missing computed profile for axis {axis_idx}")
        fits[axis_label] = generate_axis_fit(
            profile,
            width,
            spline_smoothing,
            limits,
        )
        medians[axis_label] = global_med

    return fits, medians


def apply_axis_corrections(
    full_res: da.Array,
    mask_upscaled: da.Array,
    axis_fits: dict[str, np.ndarray],
    axis_medians: dict[str, float],
    *,
    global_factor: float,
    clip_max: float = 2**16 - 1,
) -> da.Array:
    """Apply axis-specific correction curves and global scaling factors.

    Parameters
    ----------
    full_res : dask.array.Array
        Full-resolution input volume to correct.
    mask_upscaled : dask.array.Array
        Boolean mask at full resolution restricting where corrections apply.
    axis_fits : dict of str to numpy.ndarray
        Resampled correction curves for ``{"x", "y", "z"}``.
    axis_medians : dict of str to float
        Global medians associated with each axis profile.
    global_factor : float
        Target global intensity level used to scale the corrected volume.
    clip_max : float, default=2**16 - 1
        Upper bound used when clipping intensities after correction.

    Returns
    -------
    dask.array.Array
        Lazily corrected full-resolution volume.
    """

    corrected = full_res
    median_xy = float(axis_medians.get("x", 0.0))

    if median_xy != 0 and np.isfinite(median_xy):
        fit_x = axis_fits.get("x")
        if fit_x is not None:
            correction_x = fit_x.reshape(1, 1, -1)
            corrected = da.where(
                mask_upscaled, corrected / correction_x, corrected
            )

        fit_y = axis_fits.get("y")
        if fit_y is not None:
            correction_y = fit_y.reshape(1, -1, 1)
            corrected = da.where(
                mask_upscaled, corrected / correction_y, corrected
            )

        fit_z = axis_fits.get("z")
        if fit_z is not None:
            correction_z = fit_z.reshape(-1, 1, 1)
            corrected = da.where(
                mask_upscaled, corrected / correction_z, corrected
            )

        ratio = global_factor / median_xy
        _LOGGER.info(
            "Doing global correction with factor: %s and median_xy: %s, ratio = %s",
            global_factor,
            median_xy,
            ratio,
        )
        corrected = da.where(mask_upscaled, corrected * ratio, corrected)
        corrected = da.clip(corrected, 0, clip_max)
    else:
        _LOGGER.warning(
            "Skipping correction: median_xy is zero or non-finite (%s)",
            median_xy,
        )

    return corrected


def masked_axis_profile(
    volume: "np.ndarray | da.Array",
    mask: "np.ndarray | da.Array",
    axes: Iterable[int],
    *,
    smooth_sigma: "float | None" = None,
    percentile: "float | None" = None,
    min_voxels: int = 0,
) -> tuple[dict[int, np.ndarray], float]:
    """
    Measure normalized intensity profiles along specified axes using a 3D mask.

    Parameters
    ----------
    volume : np.ndarray or dask.array.Array
        3D volume containing the intensity values.
    mask : np.ndarray or dask.array.Array
        Binary mask with the same shape as ``volume``.
    axes : iterable of int
        Axes along which to evaluate the profiles.
    smooth_sigma : float, optional
        Standard deviation for optional 1D Gaussian smoothing of the profile.
    percentile : float, optional
        Percentile to use instead of the median when summarising each plane.
    min_voxels : int, optional
        Minimum number of mask voxels required for a plane to be trusted.

    Returns
    -------
    profiles : dict[int, np.ndarray]
        Mapping from axis index to profile normalised by the global masked
        median.
    global_med : float
        Global median intensity inside the mask.
    """
    if percentile is None:
        percentile = 50

    if isinstance(volume, da.Array):
        volume_np = volume.astype(np.float32).compute()
    else:
        volume_np = np.asarray(volume, dtype=np.float32)

    if isinstance(mask, da.Array):
        mask_np = mask.astype(bool).compute()
    else:
        mask_np = np.asarray(mask, dtype=bool)

    if volume_np.shape != mask_np.shape:
        raise ValueError("volume and mask must have the same shape")

    axes = tuple(dict.fromkeys(axes))
    if not axes:
        return {}, float("nan")

    axis_lengths: dict[int, int] = {}
    for axis in axes:
        if axis < 0 or axis >= mask_np.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for array of ndim {mask_np.ndim}"
            )
        axis_lengths[axis] = mask_np.shape[axis]

    masked = np.where(mask_np, volume_np, np.nan)
    global_med_value = float(np.nanpercentile(masked, percentile))

    profiles: dict[int, np.ndarray] = {}
    for axis in axes:
        reduce_axes = tuple(i for i in range(mask_np.ndim) if i != axis)
        coverage = mask_np.sum(axis=reduce_axes)
        profile = np.nanpercentile(masked, percentile, axis=reduce_axes)
        profile = np.where(np.isnan(profile), global_med_value, profile)
        if min_voxels:
            profile = np.where(coverage >= min_voxels, profile, global_med_value)
        profiles[axis] = np.asarray(profile, dtype=np.float32)

    results: dict[int, np.ndarray] = {}
    if not np.isfinite(global_med_value) or global_med_value <= 0:
        for axis, axis_len in axis_lengths.items():
            results[axis] = np.ones(axis_len, dtype=np.float32)
        return results, 0.0

    for axis, profile_np in profiles.items():
        smoothed = profile_np
        if smooth_sigma and smooth_sigma > 0:
            smoothed = gaussian_filter1d(
                smoothed, sigma=smooth_sigma, mode="nearest"
            )
        norm_profile = (smoothed / global_med_value).astype(np.float32, copy=False)
        results[axis] = norm_profile

    return results, global_med_value
