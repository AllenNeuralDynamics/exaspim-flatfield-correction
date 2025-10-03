import logging
from typing import Union

import dask.array as da
import numpy as np
from exaspim_flatfield_correction.utils.utils import resize_dask
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    label,
)
from skimage.morphology import ball, disk

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
            raise ValueError(
                "No connected components meet the specified min_size"
            )
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
    _LOGGER.debug("scale factors: %s %s %s", sz, sy, sx)

    # Upscale the SDF using affine-based resize with linear interpolation (order=1).
    sdf_upscaled = resize_dask(
        sdf_da, scale_factor=(sz, sy, sx), order=1, output_chunks=chunks
    )

    # values >= 0 are considered inside the mask.
    return sdf_upscaled >= 0
