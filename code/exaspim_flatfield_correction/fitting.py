import logging
from collections.abc import Iterable

import dask.array as da
import numpy as np
from scipy.interpolate import splev, splrep
from scipy.ndimage import gaussian_filter1d
import dask_image.ndfilters as di
import dask_image.ndmorph as ndm
from sklearn.mixture import GaussianMixture
from skimage.morphology import ball, disk

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
    percentile: "float | None" = None,
    min_voxels: int = 0,
    spline_smoothing: float = 0,
    limits_x: "tuple[float, float] | None" = None,
    limits_y: "tuple[float, float] | None" = None,
    limits_z: "tuple[float, float] | None" = None,
    weights: "np.ndarray | None" = None,
    global_med: float = None,
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
    weights : numpy.ndarray or None, optional
        Optional weighting array used for percentile calculations.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping of axis labels to fitted correction curves
    """

    axis_specs = (
        ("x", 2, full_shape[2], limits_x),
        ("y", 1, full_shape[1], limits_y),
        ("z", 0, full_shape[0], limits_z),
    )

    axis_indices = [axis_idx for _, axis_idx, _, _ in axis_specs]
    profiles = masked_axis_profile(
        volume,
        mask,
        axes=axis_indices,
        percentile=percentile,
        min_voxels=min_voxels,
        weights=weights,
        global_med_value=global_med,
    )

    fits: dict[str, np.ndarray] = {}
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

    return fits


def apply_axis_corrections(
    full_res: da.Array,
    mask_upscaled: da.Array,
    axis_fits: dict[str, np.ndarray],
    *,
    global_factor: float,
    global_med: float,
    ratio_limits: "tuple[float, float] | None" = None,
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
    global_factor : float
        Target global intensity level used to scale the corrected volume.
    ratio_limits : tuple of float or None, default=None
        Optional clamp bounds applied to the global scaling ratio.
    clip_max : float, default=2**16 - 1
        Upper bound used when clipping intensities after correction.

    Returns
    -------
    dask.array.Array
        Lazily corrected full-resolution volume.
    """
    corrected = full_res
    if global_med != 0 and np.isfinite(global_med):
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

        ratio = global_factor / global_med
        if ratio_limits is not None:
            clamped_ratio = float(
                np.clip(ratio, ratio_limits[0], ratio_limits[1])
            )
            if clamped_ratio != ratio:
                _LOGGER.info(
                    "Clamped global correction ratio from %s to %s using limits %s",
                    ratio,
                    clamped_ratio,
                    ratio_limits,
                )
            ratio = clamped_ratio
        _LOGGER.info(
            "Doing global correction with factor: %s and median_xy: %s, ratio = %s",
            global_factor,
            global_med,
            ratio,
        )
        corrected = da.where(mask_upscaled, corrected * ratio, corrected)
        corrected = da.clip(corrected, 0, clip_max)
    else:
        _LOGGER.warning(
            "Skipping correction: median_xy is zero or non-finite (%s)",
            global_med,
        )

    return corrected


def _nanpercentile_flattened(
    data: np.ndarray,
    axis: int,
    percentile: float,
    *,
    weights: "np.ndarray | None" = None,
    method: str = "linear",
) -> np.ndarray:
    """Compute percentiles along ``axis`` by flattening the remaining dims.

    This helper avoids NumPy's current limitation with weighted reductions
    over multiple axes by iterating over each slice explicitly.
    """
    if axis < 0:
        axis += data.ndim
    if axis < 0 or axis >= data.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for array of ndim {data.ndim}"
        )

    moved = np.moveaxis(data, axis, 0)
    moved_weights: "np.ndarray | None" = None
    if weights is not None:
        moved_weights = np.moveaxis(weights, axis, 0)

    result = np.empty(moved.shape[0], dtype=np.float32)

    for idx in range(moved.shape[0]):
        slice_values = moved[idx].reshape(-1)
        if moved_weights is None:
            value = np.nanpercentile(slice_values, percentile, method=method)
        else:
            slice_weights = moved_weights[idx].reshape(-1)
            value = np.nanpercentile(
                slice_values,
                percentile,
                weights=slice_weights,
                method=method,
            )
        result[idx] = value

    return result


def masked_axis_profile(
    volume: "np.ndarray | da.Array",
    mask: "np.ndarray | da.Array",
    axes: Iterable[int],
    percentile: "float | None" = None,
    min_voxels: int = 0,
    weights: "np.ndarray | None" = None,
    global_med_value: float = None,
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
    percentile : float, optional
        Percentile to use instead of the median when summarising each plane.
    min_voxels : int, optional
        Minimum number of mask voxels required for a plane to be trusted.
    weights : np.ndarray or None, optional
        Optional weighting array matching ``volume`` used for the percentile
        computation.

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

    method = "linear"
    if weights is not None:
        method = "inverted_cdf"

    masked = np.where(mask_np, volume_np, np.nan)
    if global_med_value is None:
        global_med_value = float(
            np.nanpercentile(
                masked, percentile, weights=weights, method=method
            )
        )
    _LOGGER.debug('Global median value: %s', global_med_value)

    profiles: dict[int, np.ndarray] = {}
    for axis in axes:
        reduce_axes = tuple(i for i in range(mask_np.ndim) if i != axis)
        coverage = mask_np.sum(axis=reduce_axes)
        if weights is not None:
            profile = _nanpercentile_flattened(
                masked,
                axis,
                percentile,
                weights=weights,
                method=method,
            )
        else:
            profile = np.nanpercentile(
                masked,
                percentile,
                axis=reduce_axes,
                method=method,
            )
        profile = np.where(np.isnan(profile), global_med_value, profile)
        if min_voxels:
            profile = np.where(
                coverage >= min_voxels, profile, global_med_value
            )
        profiles[axis] = np.asarray(profile, dtype=np.float32)

    results: dict[int, np.ndarray] = {}
    if not np.isfinite(global_med_value) or global_med_value <= 0:
        for axis, axis_len in axis_lengths.items():
            results[axis] = np.ones(axis_len, dtype=np.float32)
        return results, 0.0

    for axis, profile_np in profiles.items():
        norm_profile = (profile_np / global_med_value).astype(
            np.float32, copy=False
        )
        results[axis] = norm_profile

    return results


def calc_percentile_weight(
    img_da: np.ndarray | da.Array,
    bg_da: np.ndarray | da.Array,
    *,
    low_percentile: float = 50.0,
    high_percentile: float = 99.0,
    eps: float = 0.01,
    smooth_sigma: float = 1.0,
    start_frac: float = 0,
    nu: float = 1.0,
) -> da.Array:
    """
    Compute foreground weights using a generalised logistic (Richards) curve.

    This version gives you more asymmetric control: `start_frac` picks where
    the ramp begins (relative to low→high), and `nu` adjusts skew/inflection bias.
    The top “knee” is anchored near (p_high, 1−eps).

    Parameters
    ----------
    img_da : np.ndarray or dask.array.Array
        Background-subtracted image volume.
    bg_da : np.ndarray or dask.array.Array
        Voxels representative of the background distribution.
    low_percentile : float
        Lower percentile (maps to weight ≈ 0 region).
    high_percentile : float
        Upper percentile (maps close to weight = 1).
    eps : float
        A small positive parameter so the extremes don’t hit exactly 0 or 1.
    smooth_sigma : float
        Gaussian smoothing sigma on the resulting weight map.
    start_frac : float (in [0,1))
        Fraction along (low→high) where the ramp should **begin** in earnest.
    nu : float (> 0)
        Skew / asymmetry parameter of the generalised logistic (nu = 1 → standard logistic).

    Returns
    -------
    dask.array.Array
        Weights in [0,1], softly ramping from “background-like” to foreground.
    """

    img = da.asarray(img_da, dtype=np.float32)

    # Compute background percentiles
    percentiles = np.percentile(bg_da, [low_percentile, high_percentile])
    p_low = float(percentiles[0])
    p_high = float(percentiles[1])

    if not np.isfinite(p_low) or not np.isfinite(p_high):
        raise ValueError("Background percentiles must be finite values")

    if p_high < p_low:
        # swap to ensure p_low < p_high
        p_low, p_high = p_high, p_low

    eps = float(np.clip(eps, 1e-6, 0.49))
    start_frac = float(np.clip(start_frac, 0.0, 0.999))
    nu = float(np.clip(nu, 1e-6, None))

    delta = p_high - p_low
    if delta <= 0:
        # degenerate: everything is basically the same
        weights = da.where(img > p_low, 1.0, 0.0).astype(np.float32)
    else:
        # Normalize intensity to t in [0,1]
        def _norm(block: np.ndarray) -> np.ndarray:
            return np.clip((block - p_low) / delta, 0.0, 1.0)

        t = da.map_blocks(_norm, img, dtype=np.float32)

        # Anchors:
        #   At t = start_frac, weight = eps
        #   At t = 1.0, weight = 1 - eps
        y1 = eps
        y2 = 1.0 - eps

        # According to Richards/generalised logistic with A=0, K=1, C=1:
        #   f(t) = [1 + Q * exp(-B t)]^(-1/nu)
        # Solve for B and Q:
        #   (1 + Q exp(-B t1)) = y1^{-ν}
        #   (1 + Q exp(-B * 1))  = y2^{-ν}
        A_s = y1 ** (-nu) - 1.0
        A_h = y2 ** (-nu) - 1.0
        # Avoid divide by zero
        B = np.log(A_s / A_h) / (1.0 - start_frac)
        Q = float(A_h * np.exp(B))

        # Logistic / Richards block
        def _richards(block: np.ndarray) -> np.ndarray:
            # block is normalized t
            # compute exponent part
            z = -B * np.clip(block, 0.0, 1.0)
            # compute (1 + Q exp(z))^(−1/nu); z is <= 0 so exp(z) is stable
            return (1.0 / (1.0 + Q * np.exp(z))) ** (1.0 / nu)

        weights = da.map_blocks(_richards, t, dtype=np.float32)

    # Optional smoothing
    if smooth_sigma and smooth_sigma > 0:
        weights = di.gaussian_filter(weights, sigma=smooth_sigma)

    # Clip to [0,1]
    weights = da.clip(weights, 0.0, 1.0).astype(np.float32)
    return weights


def compute_simple_features_dask(
    x: np.ndarray | da.Array,
    *,
    sigma_mean: float = 1.5,
    sigma_log: float = 1.0,
    sigma_grad: float = 1.0,
) -> list[da.Array]:
    """Compute a small bank of local features in a lazy fashion.

    Parameters
    ----------
    x : numpy.ndarray or dask.array.Array
        Input image volume.
    sigma_mean : float, default=1.5
        Gaussian sigma used when computing the local mean and variance.
    sigma_log : float, default=1.0
        Sigma for the Laplacian-of-Gaussian feature.
    sigma_grad : float, default=1.0
        Sigma for the Gaussian gradient magnitude feature.

    Returns
    -------
    list of dask.array.Array
        Lazy feature volumes ``[intensity, local mean, local std, gradient
        magnitude, -LoG]``.
    """
    # Local mean & std via Gaussian filtering (E[x^2] - (E[x])^2)
    m = di.gaussian_filter(x, sigma=sigma_mean)
    m2 = di.gaussian_filter(x * x, sigma=sigma_mean)
    var = da.maximum(m2 - m * m, 0.0).astype(np.float32)
    std = da.sqrt(var)

    # Edge/texture cues
    grad_mag = di.gaussian_gradient_magnitude(x, sigma=sigma_grad).astype(
        np.float32
    )
    neg_log = (-di.gaussian_laplace(x, sigma=sigma_log)).astype(np.float32)

    return [x, m, std, grad_mag, neg_log]


def _sample_rows_from_dask_stack(
    feats: list[da.Array],
    linear_idx: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gather a subset of feature vectors from lazily computed arrays.

    Parameters
    ----------
    feats : list of dask.array.Array
        Feature volumes produced by :func:`compute_simple_features_dask`.
    linear_idx : numpy.ndarray
        Flat indices identifying candidate voxels.
    max_rows : int
        Maximum number of rows to sample from ``linear_idx``.
    rng : numpy.random.Generator
        Random generator used to subsample indices.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_samples, n_features)`` containing finite feature
        vectors.

    Raises
    ------
    ValueError
        If no finite feature vectors can be assembled.
    """
    if linear_idx.size > max_rows:
        linear_idx = rng.choice(linear_idx, size=max_rows, replace=False)

    # Advanced indexing gathers only the needed rows from each feature
    cols = [
        da.take(f.ravel(), linear_idx) for f in feats
    ]  # list of dask 1D arrays
    X = da.stack(cols, axis=1).compute()  # small NumPy (n_sel, n_feats)

    finite = np.isfinite(X).all(axis=1)
    if not np.any(finite):
        raise ValueError("No finite feature vectors in sampled rows")
    return X[finite].astype(np.float64, copy=False)


def calc_gmm_prob(
    img_da: np.ndarray | da.Array,
    mask_da: np.ndarray,
    bg_da: np.ndarray | da.Array,
    n_components_fg: int = 3,
    n_components_bg: int = 3,
    max_samples_fg: int = 500_000,
    max_samples_bg: int | None = 500_000,
    prior_fg: float = 0.5,
    reg_covar: float = 1e-6,
    n_init: int = 3,
    random_state: int | None = 0,
    sigma_mean: float = 1.0,
    sigma_log: float = 1.0,
    sigma_grad: float = 1.0,
    erosion_radius: int = 1,
) -> da.Array:
    """Estimate foreground probability with paired Gaussian mixture models.

    Parameters
    ----------
    img_da : numpy.ndarray or dask.array.Array
        Image volume whose foreground probability is estimated.
    mask_da : numpy.ndarray
        Binary mask highlighting confident foreground voxels.
    bg_da : numpy.ndarray or dask.array.Array
        Background reference volume used to learn background appearance.
    n_components_fg : int, default=3
        Number of mixture components for the foreground model.
    n_components_bg : int, default=3
        Number of mixture components for the background model.
    max_samples_fg : int, default=500000
        Maximum number of foreground voxels used for model fitting.
    max_samples_bg : int or None, default=500000
        Maximum number of background voxels sampled; use ``None`` to take the
        full background volume.
    prior_fg : float, default=0.5
        Foreground prior probability in the Bayes classifier.
    reg_covar : float, default=1e-6
        Covariance regularisation for ``GaussianMixture``.
    n_init : int, default=3
        Number of mixture initialisations.
    random_state : int or None, default=0
        Seed used by the random generator.
    sigma_mean : float, default=1.0
        Sigma for the local mean feature.
    sigma_log : float, default=1.0
        Sigma for the Laplacian-of-Gaussian feature.
    sigma_grad : float, default=1.0
        Sigma for the gradient magnitude feature.
    erosion_radius : int, default=1
        Radius of the binary erosion applied to ``mask_da`` before sampling.

    Returns
    -------
    dask.array.Array
        Lazy array of foreground probabilities shaped like ``img_da``.

    Raises
    ------
    ValueError
        If ``prior_fg`` is outside ``(0, 1)`` or sampling constraints are
        incompatible with the provided data.
    """
    if not (0.0 < prior_fg < 1.0):
        raise ValueError("prior_fg must be in (0, 1)")

    # ---- Features (lazy, via dask-image) ----
    feats_img = compute_simple_features_dask(
        img_da,
        sigma_mean=sigma_mean,
        sigma_log=sigma_log,
        sigma_grad=sigma_grad,
    )
    feats_bg = compute_simple_features_dask(
        bg_da,
        sigma_mean=sigma_mean,
        sigma_log=sigma_log,
        sigma_grad=sigma_grad,
    )
    # ---- Subsample training rows from Dask (materialize small NumPy arrays) ----
    rng = np.random.default_rng(random_state)

    if erosion_radius > 0:
        if mask_da.ndim == 3:
            structure = ball(erosion_radius)
        elif mask_da.ndim == 2:
            structure = disk(erosion_radius)
        else:
            raise ValueError("Mask must be 2D or 3D for erosion")
        _LOGGER.debug(
            "Applying binary erosion to mask with radius %s", erosion_radius
        )
        mask_eroded = ndm.binary_erosion(mask_da, structure=structure)
    else:
        mask_eroded = mask_da

    fg_lin_idx = da.flatnonzero(mask_eroded.ravel()).compute()
    _LOGGER.debug("Sampling foreground feature vectors")
    X_fg = _sample_rows_from_dask_stack(
        feats_img, fg_lin_idx, max_samples_fg, rng
    )

    # background: uniform sample across entire background volume
    bg_size = int(np.prod(bg_da.shape))

    if max_samples_bg is not None:
        if max_samples_bg <= 0:
            raise ValueError("max_samples_bg must be positive or None")
        sample_cap = min(max_samples_bg, bg_size)
        if sample_cap >= bg_size:
            bg_lin_idx = np.arange(bg_size, dtype=np.int64)
        else:
            bg_lin_idx = rng.choice(bg_size, size=sample_cap, replace=False)
    else:
        sample_cap = bg_size
        _LOGGER.debug("Sampling full background volume (cap=%s)", sample_cap)
        bg_lin_idx = np.arange(bg_size, dtype=np.int64)
    _LOGGER.debug("Sampling background feature vectors")
    X_bg = _sample_rows_from_dask_stack(feats_bg, bg_lin_idx, sample_cap, rng)

    # ---- Standardize by combined training stats ----
    train_stack = np.vstack([X_fg, X_bg])
    mean_vec = train_stack.mean(axis=0)
    std_vec = train_stack.std(axis=0)
    std_vec[std_vec < 1e-6] = 1.0

    # ---- Fit GMMs ----
    fg_gmm = GaussianMixture(
        n_components=n_components_fg,
        covariance_type="full",
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    ).fit((X_fg - mean_vec) / std_vec)

    bg_gmm = GaussianMixture(
        n_components=n_components_bg,
        covariance_type="full",
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    ).fit((X_bg - mean_vec) / std_vec)

    # ---- Stack features lazily for blockwise scoring: (..., F) ----
    feats_stack = da.stack(feats_img, axis=-1).astype(
        np.float32
    )  # same chunks as img

    # ---- Blockwise scoring (mask-aware) ----
    log_pi_fg = float(np.log(prior_fg))
    log_pi_bg = float(np.log(1.0 - prior_fg))

    # --- change _score_block to accept a 4D mask with a singleton last axis ---
    def _score_block(
        block_feats: np.ndarray,
        block_mask: np.ndarray,
        fg_gmm: GaussianMixture,
        bg_gmm: GaussianMixture,
        mean_vec: np.ndarray,
        std_vec: np.ndarray,
        log_pi_fg: float,
        log_pi_bg: float,
    ) -> np.ndarray:

        # If mask arrived as (..., 1), squeeze it back to (...)
        if block_mask.ndim == block_feats.ndim:
            block_mask = np.squeeze(block_mask, axis=-1)

        out = np.zeros(block_mask.shape, dtype=np.float32)
        m = block_mask.astype(bool, copy=False)
        if not np.any(m):
            return out

        X = (
            block_feats[m]
            .reshape(-1, block_feats.shape[-1])
            .astype(np.float64, copy=False)
        )
        X = (X - mean_vec) / std_vec
        finite = np.isfinite(X).all(axis=1)

        if np.any(finite):
            Xz = X[finite]
            lg_f = fg_gmm.score_samples(Xz) + log_pi_fg
            lg_b = bg_gmm.score_samples(Xz) + log_pi_bg
            prob = np.exp(lg_f - np.logaddexp(lg_f, lg_b)).astype(
                np.float32, copy=False
            )

            tmp = np.zeros(X.shape[0], dtype=np.float32)
            tmp[finite] = prob
            out[m] = tmp

        return out

    probs = da.map_blocks(
        _score_block,
        feats_stack,  # 4D blocks
        mask_da[..., None],  # 4D blocks with last axis = 1
        fg_gmm=fg_gmm,
        bg_gmm=bg_gmm,
        mean_vec=mean_vec,
        std_vec=std_vec,
        log_pi_fg=log_pi_fg,
        log_pi_bg=log_pi_bg,
        dtype=np.float32,
        chunks=img_da.chunks,  # output is 3D like image
        drop_axis=(-1,),  # drop the feature axis in the output
    )

    return probs
