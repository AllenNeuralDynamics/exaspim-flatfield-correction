import numpy as np
import dask.array as da
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splrep, splev
import logging

_LOGGER = logging.getLogger(__name__)


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


def generate_axis_fit(
    profile: "np.ndarray | da.Array",
    new_width: int,
    smoothing: float,
    limits: "tuple[float, float] | None" = None,
) -> np.ndarray:
    """Scale a 1D normalized profile to a new width with optional clipping."""
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
    """Compute normalized profiles and spline fits along x, y, and z axes."""

    axis_specs = (
        ("x", 2, full_shape[2], limits_x),
        ("y", 1, full_shape[1], limits_y),
        ("z", 0, full_shape[0], limits_z),
    )

    fits: dict[str, np.ndarray] = {}
    medians: dict[str, float] = {}

    for axis_label, axis_idx, width, limits in axis_specs:
        profile, median = masked_axis_profile(
            volume,
            mask,
            axis=axis_idx,
            smooth_sigma=smooth_sigma,
            percentile=percentile,
            min_voxels=min_voxels,
        )
        fits[axis_label] = generate_axis_fit(
            profile,
            width,
            spline_smoothing,
            limits,
        )
        medians[axis_label] = median

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
    """Apply per-axis correction profiles and global scaling to a volume."""

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
    axis: int,
    *,
    smooth_sigma: "float | None" = None,
    percentile: "float | None" = None,
    min_voxels: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Measure a normalized intensity profile along an axis using a 3D mask.

    Parameters
    ----------
    volume : np.ndarray or dask.array.Array
        3D volume containing the intensity values.
    mask : np.ndarray or dask.array.Array
        Binary mask with the same shape as ``volume``.
    axis : int
        Axis along which to evaluate the profile.
    smooth_sigma : float, optional
        Standard deviation for optional 1D Gaussian smoothing of the profile.
    percentile : float, optional
        Percentile to use instead of the median when summarising each plane.
    min_voxels : int, optional
        Minimum number of mask voxels required for a plane to be trusted.

    Returns
    -------
    norm_profile : np.ndarray
        Profile normalised by the global masked median.
    global_med : float
        Global median intensity inside the mask.
    """
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

    masked = np.where(mask_np, volume_np, np.nan)

    reduce_axes = tuple(i for i in range(mask_np.ndim) if i != axis)
    coverage = mask_np.sum(axis=reduce_axes)

    if percentile is None:
        percentile = 50

    profile = np.nanpercentile(masked, percentile, axis=reduce_axes)

    global_med = float(np.nanpercentile(masked, percentile))
    if not np.isfinite(global_med) or global_med <= 0:
        return np.ones(mask_np.shape[axis], dtype=np.float32), 0.0

    profile = np.where(np.isnan(profile), global_med, profile)
    if min_voxels:
        profile = np.where(coverage >= min_voxels, profile, global_med)

    if smooth_sigma and smooth_sigma > 0:
        profile = gaussian_filter1d(
            profile, sigma=smooth_sigma, mode="nearest"
        )

    norm_profile = (profile / global_med).astype(np.float32, copy=False)
    return norm_profile, global_med
