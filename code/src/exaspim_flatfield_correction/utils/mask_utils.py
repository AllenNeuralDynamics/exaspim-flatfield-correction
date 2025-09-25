import logging
from typing import Union

import dask.array as da
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    label,
)
import dask_image.ndfilters as di  
import dask_image.ndmorph as ndm
from skimage.morphology import ball, disk
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel

from exaspim_flatfield_correction.utils.utils import resize_dask

_LOGGER = logging.getLogger(__name__)


def get_mask(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int = 2,
    min_size: int | None = None,
    k_largest: int | None = None,
) -> np.ndarray:
    """
    Create a binary mask for a fluorescent 2D or 3D image of a mouse brain.

    By default, this performs morphological cleanup (closing + fill holes)
    without any size-based filtering. To filter by size, set `min_size` and/or
    `k_largest`.

    Parameters
    ----------
    volume : numpy.ndarray
        2D or 3D grayscale image (background subtracted).
    threshold : float
        Manual threshold value for segmentation.
    closing_radius : int, optional
        Radius for the structuring element used in binary closing.
        Default is 2.
    min_size : int, optional
        Minimum size (in pixels/voxels) for connected components to be retained.
        If None and `k_largest` is also None, no size filtering is applied.
    k_largest : int, optional
        If provided, retain the K largest connected components. If `min_size` is
        also provided, select the K largest among components meeting `min_size`.

    Returns
    -------
    final_mask : numpy.ndarray, bool
        A binary mask of the brain tissue after morphological cleanup and
        optional size filtering.
    """
    return process_mask(
        volume > threshold,
        closing_radius=closing_radius,
        min_size=min_size,
        k_largest=k_largest,
    )


def process_mask(
    mask: np.ndarray,
    closing_radius: int = 2,
    min_size: int | None = None,
    k_largest: int | None = None,
) -> np.ndarray:
    """
    Process a binary mask with morphological cleanup and optional component
    size filtering.

    Parameters
    ----------
    mask : numpy.ndarray, bool
        2D or 3D binary mask (boolean or 0/1 values).
    closing_radius : int, optional
        Radius for the structuring element used in binary closing.
        Default is 2.
    min_size : int, optional
        Minimum size (in pixels/voxels) for connected components to be retained.
        If None and `k_largest` is also None, no size filtering is applied.
    k_largest : int, optional
        If provided, retain the K largest connected components. If `min_size` is
        also provided, select the K largest among components meeting `min_size`.

    Returns
    -------
    final_mask : numpy.ndarray, bool
        A binary mask after morphological cleanup and optional size filtering.
    """
    # Choose the appropriate structuring element based on image dimensions.
    if mask.ndim == 3:
        selem = ball(closing_radius)
    elif mask.ndim == 2:
        selem = disk(closing_radius)
    else:
        raise ValueError("Input volume must be 2D or 3D.")

    mask = binary_fill_holes(binary_closing(mask, selem))

    return size_filter(mask, min_size=min_size, k_largest=k_largest)


def size_filter(
    mask: np.ndarray,
    min_size: int | None = None,
    k_largest: int | None = None,
) -> np.ndarray:
    """
    Optionally filter connected components by size and/or keep the K largest.

    Rules:
    - If min_size is None and k_largest is None: return mask unchanged.
    - If min_size is set and no components satisfy it: raise ValueError.
    - If k_largest is set and min_size is also set: keep the top-K among
      components >= min_size; if none satisfy, raise ValueError.
    - If k_largest is set and min_size is None: keep the top-K among all
      components (ignoring background).
    - If only min_size is set: keep all components >= min_size; if none, raise.
    """
    # No size filtering requested
    if min_size is None and k_largest is None:
        return mask

    # Label connected components in the filled mask.
    labeled_array, num_features = label(mask)
    _LOGGER.debug(f"Number of connected features: {num_features}")
    if num_features == 0:
        raise ValueError("No connected components found in the image.")

    if num_features < 2**16:
        labeled_array = labeled_array.astype(np.uint16)

    # Calculate sizes for each component (ignore background label 0).
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # ignore background

    if k_largest is not None:
        if k_largest <= 0:
            raise ValueError("k_largest must be a positive integer")

        # Choose candidates by min_size if provided
        if min_size is not None:
            candidate_labels = np.where(component_sizes >= min_size)[0]
            candidate_labels = candidate_labels[candidate_labels != 0]
            if candidate_labels.size == 0:
                raise ValueError(
                    "No connected components meet the specified min_size"
                )
        else:
            candidate_labels = np.where(component_sizes > 0)[0]

        candidate_sizes = component_sizes[candidate_labels]
        order = np.argsort(candidate_sizes)[::-1]
        keep_labels = candidate_labels[order][:k_largest]
        final_mask = np.isin(labeled_array, keep_labels)
    else:
        # Only min_size is set
        valid_labels = np.where(component_sizes >= min_size)[0]
        valid_labels = valid_labels[valid_labels != 0]
        if len(valid_labels) == 0:
            raise ValueError("No connected components meet the specified min_size")
        final_mask = np.isin(labeled_array, valid_labels)

    return final_mask


def upscale_mask_nearest(
    mask: np.ndarray,
    new_shape: tuple[int, int, int],
    chunks: Union[int, tuple, str] = "auto",
) -> da.Array:
    """
    Upscales a 3D binary mask to a new shape using nearest-neighbor
    interpolation with an affine transform via Dask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (boolean or 0/1 values).
    new_shape : tuple of ints
        Desired shape (depth, height, width) for the upscaled mask.
        Assumes uniform scaling.
    chunks : int, tuple, or 'auto', optional
        Chunk size to use for the Dask array. Default is 'auto'.

    Returns
    -------
    dask.array.Array
        The upscaled mask as a Dask array.
    """
    # Compute anisotropic per-axis scale factors
    sz = new_shape[0] / mask.shape[0]
    sy = new_shape[1] / mask.shape[1]
    sx = new_shape[2] / mask.shape[2]

    # Use nearest-neighbor interpolation (order=0)
    upscaled = resize_dask(
        mask, scale_factor=(sz, sy, sx), order=0, output_chunks=chunks
    )

    return upscaled


def upscale_mask_edt(
    mask: np.ndarray,
    new_shape: tuple[int, int, int],
    chunks: "int | tuple | str" = "auto",
) -> da.Array:
    """
    Upscales a 3D binary mask using a distance-transform approach with Dask
    and an affine transformation for resizing.

    The procedure is:
      1. Compute the signed distance field (SDF) on the original numpy mask.
      2. Convert the SDF to a Dask array with desired chunking.
      3. Upscale the SDF using the affine transformation based resize.
      4. Threshold at 0 to recover a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (boolean or 0/1 values).
    new_shape : tuple of ints
        Desired shape (depth, height, width) for the upscaled mask.
        Assumes uniform scaling.
    chunks : int, tuple, or 'auto'
        Chunk size to use for the Dask array.

    Returns
    -------
    dask.array
        Dask array containing the upscaled binary mask.
    """
    # Ensure the mask is boolean.
    mask_bool = mask.astype(bool)

    # Compute distance transforms (inside and outside).
    dt_inside = distance_transform_edt(mask_bool)
    dt_outside = distance_transform_edt(~mask_bool)

    # Compute the signed distance field (SDF): positive inside,
    # negative outside.
    sdf = dt_inside - dt_outside

    # Convert the SDF to a Dask array.
    sdf_da = da.from_array(sdf.astype(np.float32), chunks=chunks)

    # Compute anisotropic per-axis scale factors
    sz = new_shape[0] / mask.shape[0]
    sy = new_shape[1] / mask.shape[1]
    sx = new_shape[2] / mask.shape[2]
    print("scale factors:", sz, sy, sx)

    # Upscale the SDF using affine-based resize with linear interpolation (order=1).
    sdf_upscaled = resize_dask(
        sdf_da, scale_factor=(sz, sy, sx), order=1, output_chunks=chunks
    )

    # values >= 0 are considered inside the mask.
    return sdf_upscaled >= 0


def gmm_probability_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_volume: np.ndarray,
    n_components_fg: int = 4,
    n_components_bg: int = 2,
    max_samples_fg: int = 500_000,
    max_samples_bg: int = 500_000,
    batch_size: int = 50_000,
    prior_fg: float = 0.5,
    reg_covar: float = 1e-6,
    n_init: int = 3,
    random_state: int | None = 0,
    erosion_radius: int = 1,
) -> np.ndarray:
    """Estimate P(foreground | intensity) inside `mask` using two class-conditional GMMs.

    Parameters
    ----------
    image : np.ndarray
        Raw intensity volume. Must have the same shape as `mask`.
    mask : np.ndarray
        Boolean or uint8 mask defining the foreground region to score.
    background_volume : np.ndarray
        Background-only intensities (any shape); used to fit the background GMM.
    n_components_fg, n_components_bg : int
        Mixture component counts for the foreground/background models.
    max_samples_fg, max_samples_bg : int
        Cap on samples used to fit each model (uniform subsampling if exceeded).
    batch_size : int
        Number of voxels per scoring batch to limit peak memory use.
    prior_fg : float
        Prior probability of foreground in (0,1); balances precision/recall.
    reg_covar : float
        Small diagonal added to covariances for numerical stability.
    n_init : int
        Number of EM initializations (best init kept).
    random_state : int | None
        Seed for reproducibility.
    erosion_radius : int
        Radius of the structuring element used to erode the foreground mask
        before fitting the foreground GMM. Set to 0 to disable erosion.

    Returns
    -------
    np.ndarray
        Float32 array of same shape as `image` with P(foreground) inside `mask`
        and zeros outside.
    """
    if image.shape != mask.shape:
        raise ValueError("image and mask must share the same shape")
    if not (0.0 < prior_fg < 1.0):
        raise ValueError("prior_fg must be in (0, 1)")

    # Foreground samples (from inside mask)
    mask_bool = mask.astype(bool, copy=False)
    if not np.any(mask_bool):
        raise ValueError("Mask does not contain any foreground voxels")

    fg_mask = mask_bool
    if erosion_radius > 0:
        if mask.ndim == 3:
            structure = ball(erosion_radius)
        elif mask.ndim == 2:
            structure = disk(erosion_radius)
        else:
            raise ValueError("Mask must be 2D or 3D for erosion")

        eroded_mask = binary_erosion(mask_bool, structure=structure)
        if np.any(eroded_mask):
            fg_mask = eroded_mask
        else:
            _LOGGER.warning(
                "Eroded mask is empty; falling back to the original mask for GMM fitting"
            )

    flat = image.ravel()
    fg_idx_all = np.flatnonzero(fg_mask.ravel())
    fg_finite = np.isfinite(flat[fg_idx_all])
    if not np.any(fg_finite):
        raise ValueError("No finite intensities inside mask for GMM fitting")
    fg_idx_all = fg_idx_all[fg_finite]

    rng = np.random.default_rng(random_state)
    if fg_idx_all.size > max_samples_fg:
        fg_idx = rng.choice(fg_idx_all, size=max_samples_fg, replace=False)
    else:
        fg_idx = fg_idx_all
    X_fg = flat[fg_idx].astype(np.float32, copy=False).reshape(-1, 1)

    # Background samples (from provided background volume; shape can differ)
    bg_vals = np.asarray(background_volume, dtype=np.float32).ravel()
    bg_vals = bg_vals[np.isfinite(bg_vals)]
    if bg_vals.size == 0:
        raise ValueError("Background volume has no finite voxels for GMM fitting")
    if bg_vals.size > max_samples_bg:
        bg_vals = rng.choice(bg_vals, size=max_samples_bg, replace=False)
    X_bg = bg_vals.reshape(-1, 1)

    # Fit class-conditional GMMs
    fg_gmm = GaussianMixture(
        n_components=n_components_fg,
        covariance_type="full",
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    ).fit(X_fg)

    bg_gmm = GaussianMixture(
        n_components=n_components_bg,
        covariance_type="full",
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    ).fit(X_bg)

    # Bayes posterior, scored only inside the mask
    out = np.zeros(flat.size, dtype=np.float32)
    score_idx = np.flatnonzero(mask_bool.ravel())
    log_pi_fg = np.log(prior_fg)
    log_pi_bg = np.log(1.0 - prior_fg)

    for start in range(0, score_idx.size, batch_size):
        bidx = score_idx[start : start + batch_size]
        x = flat[bidx].astype(np.float32, copy=False)
        finite = np.isfinite(x)
        if not np.any(finite):
            continue
        x2d = x[finite].reshape(-1, 1)

        log_fg = fg_gmm.score_samples(x2d) + log_pi_fg
        log_bg = bg_gmm.score_samples(x2d) + log_pi_bg
        denom = np.logaddexp(log_fg, log_bg)
        prob = np.exp(log_fg - denom)

        out[bidx[finite]] = prob.astype(np.float32, copy=False)

    return out.reshape(image.shape)



def project_probability_mask(
    probability_volume: np.ndarray,
    axis: int,
    threshold: float,
    min_size: int | None = None,
) -> np.ndarray:
    """Project a probability volume along one axis and threshold the maximum.

    Parameters
    ----------
    probability_volume : np.ndarray
        Soft mask (values in [0, 1]).
    axis : int
        Axis along which to compute the projection.
    threshold : float
        Minimum probability to mark a pixel as foreground in the projection.

    Returns
    -------
    np.ndarray
        Binary projection mask with optional small-component removal.
    """

    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be within (0, 1]")

    max_proj = probability_volume.max(axis=axis)
    mask_2d = max_proj >= threshold

    if min_size is not None:
        mask_2d = size_filter(mask_2d, min_size=min_size)

    return mask_2d

def compute_simple_features_dask(
    x: np.ndarray | da.Array,
    *,
    sigma_mean: float = 1.5,
    sigma_log: float = 1.0,
    sigma_grad: float = 1.0,
    chunks: tuple[int, ...] | int | str = "auto",
) -> list[da.Array]:
    """
    Lazily compute 5 features as Dask arrays using dask-image:
      [ intensity, local mean, local std, gaussian gradient magnitude, -LoG ]
    """
    # Local mean & std via Gaussian filtering (E[x^2] - (E[x])^2)
    m  = di.gaussian_filter(x, sigma=sigma_mean)
    m2 = di.gaussian_filter(x * x, sigma=sigma_mean)
    var = da.maximum(m2 - m * m, 0.0).astype(np.float32)
    std = da.sqrt(var)

    # Edge/texture cues
    grad_mag = di.gaussian_gradient_magnitude(x, sigma=sigma_grad).astype(np.float32)
    neg_log  = (-di.gaussian_laplace(x, sigma=sigma_log)).astype(np.float32)

    return [x, m, std, grad_mag, neg_log]

# -------------------- Training utilities --------------------

def _sample_rows_from_dask_stack(
    feats: list[da.Array],
    linear_idx: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gather up to max_rows rows (feature vectors) from dask feature arrays."""
    if linear_idx.size > max_rows:
        linear_idx = rng.choice(linear_idx, size=max_rows, replace=False)

    # Advanced indexing gathers only the needed rows from each feature
    cols = [da.take(f.ravel(), linear_idx) for f in feats]   # list of dask 1D arrays
    X = da.stack(cols, axis=1).compute()                     # small NumPy (n_sel, n_feats)

    finite = np.isfinite(X).all(axis=1)
    if not np.any(finite):
        raise ValueError("No finite feature vectors in sampled rows")
    return X[finite].astype(np.float64, copy=False)

# -------------------- Main: Dask 2-GMM with feature vectors --------------------

def gmm_probability_mask_features(
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
    """
    Memory-friendly 2-GMM classifier using dask-image filters.
    Returns a Dask array of P(foreground) you can `.compute()` or `.persist()`.
    The provided mask is eroded (unless `erosion_radius` is 0) before fitting
    the foreground model to focus on high-confidence voxels. Output shape ==
    image.shape, dtype float32. Set `max_samples_bg` to None to use the entire
    background volume without random sampling.
    """
    if not (0.0 < prior_fg < 1.0):
        raise ValueError("prior_fg must be in (0, 1)")

    # ---- Features (lazy, via dask-image) ----
    feats_img = compute_simple_features_dask(
        img_da, sigma_mean=sigma_mean, sigma_log=sigma_log, sigma_grad=sigma_grad, chunks=img_da.chunks
    )
    feats_bg  = compute_simple_features_dask(
        bg_da,  sigma_mean=sigma_mean, sigma_log=sigma_log, sigma_grad=sigma_grad, chunks=bg_da.chunks
    )
    n_feats = len(feats_img)  # 5

    # ---- Subsample training rows from Dask (materialize small NumPy arrays) ----
    rng = np.random.default_rng(random_state)
    
    if erosion_radius > 0:
        if mask_da.ndim == 3:
            structure = ball(erosion_radius)
        elif mask_da.ndim == 2:
            structure = disk(erosion_radius)
        else:
            raise ValueError("Mask must be 2D or 3D for erosion")
        print("eroding mask")
        mask_da = ndm.binary_erosion(mask_da, structure=structure)

    fg_lin_idx = da.flatnonzero(mask_da.ravel()).compute()
    print("sampling fg features")
    X_fg = _sample_rows_from_dask_stack(feats_img, fg_lin_idx, max_samples_fg, rng)

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
        print("cap ", sample_cap)
        bg_lin_idx = np.arange(bg_size, dtype=np.int64)
    print("sampling bg features")
    X_bg = _sample_rows_from_dask_stack(feats_bg, bg_lin_idx, sample_cap, rng)

    # ---- Standardize by combined training stats ----
    train_stack = np.vstack([X_fg, X_bg])
    mean_vec = train_stack.mean(axis=0)
    std_vec  = train_stack.std(axis=0)
    std_vec[std_vec < 1e-6] = 1.0

    # ---- Fit GMMs ----
    fg_gmm = GaussianMixture(
        n_components=n_components_fg, covariance_type="full",
        reg_covar=reg_covar, n_init=n_init, random_state=random_state
    ).fit((X_fg - mean_vec) / std_vec)

    bg_gmm = GaussianMixture(
        n_components=n_components_bg, covariance_type="full",
        reg_covar=reg_covar, n_init=n_init, random_state=random_state
    ).fit((X_bg - mean_vec) / std_vec)

    # ---- Stack features lazily for blockwise scoring: (..., F) ----
    feats_stack = da.stack(feats_img, axis=-1).astype(np.float32)  # same chunks as img

    # ---- Blockwise scoring (mask-aware) ----
    log_pi_fg = float(np.log(prior_fg))
    log_pi_bg = float(np.log(1.0 - prior_fg))

    # --- change _score_block to accept a 4D mask with a singleton last axis ---
    def _score_block(block_feats: np.ndarray,
                    block_mask: np.ndarray,
                    fg_gmm: GaussianMixture,
                    bg_gmm: GaussianMixture,
                    mean_vec: np.ndarray,
                    std_vec: np.ndarray,
                    log_pi_fg: float,
                    log_pi_bg: float) -> np.ndarray:

        # If mask arrived as (..., 1), squeeze it back to (...)
        if block_mask.ndim == block_feats.ndim:
            block_mask = np.squeeze(block_mask, axis=-1)

        out = np.zeros(block_mask.shape, dtype=np.float32)
        m = block_mask.astype(bool, copy=False)
        if not np.any(m):
            return out

        X = block_feats[m].reshape(-1, block_feats.shape[-1]).astype(np.float64, copy=False)
        X = (X - mean_vec) / std_vec
        finite = np.isfinite(X).all(axis=1)

        if np.any(finite):
            Xz   = X[finite]
            lg_f = fg_gmm.score_samples(Xz) + log_pi_fg
            lg_b = bg_gmm.score_samples(Xz) + log_pi_bg
            prob = np.exp(lg_f - np.logaddexp(lg_f, lg_b)).astype(np.float32, copy=False)

            tmp = np.zeros(X.shape[0], dtype=np.float32)
            tmp[finite] = prob
            out[m] = tmp

        return out

    probs = da.map_blocks(
        _score_block,
        feats_stack,                  # 4D blocks
        mask_da[..., None],           # 4D blocks with last axis = 1
        fg_gmm=fg_gmm,
        bg_gmm=bg_gmm,
        mean_vec=mean_vec,
        std_vec=std_vec,
        log_pi_fg=log_pi_fg,
        log_pi_bg=log_pi_bg,
        dtype=np.float32,
        chunks=img_da.chunks,         # output is 3D like image
        drop_axis=(-1,),              # drop the feature axis in the output
    )

    return probs
