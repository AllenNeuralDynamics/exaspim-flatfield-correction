import numpy as np
import dask.array as da
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_fill_holes, binary_closing
from scipy.ndimage import label
from skimage.morphology import ball, disk
import logging
from typing import Union

from exaspim_flatfield_correction.utils.utils import resize_dask

_LOGGER = logging.getLogger(__name__)


def get_mask(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int = 2,
    min_size: int = None,
) -> np.ndarray:
    """
    Create a binary mask for a fluorescent 2D or 3D image of a mouse brain.

    By default, only the largest connected component is retained
    (assumed to be the brain tissue).
    Alternatively, if a minimum component size (min_size) is provided
    (e.g., 10000 pixels),
    then all connected components above this size are kept.
    If no components meet the min_size,
    the function falls back to retaining only the largest connected component.

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
        Minimum size (in pixels) for connected components to be retained.
        If None (default), only the largest connected component is retained.

    Returns
    -------
    final_mask : numpy.ndarray, bool
        A binary mask of the brain tissue containing either the largest
        connected component or all connected components above the
        specified min_size (falling back to the largest component
        if none meet the min_size).
    """
    return process_mask(
        volume > threshold, closing_radius=closing_radius, min_size=min_size
    )


def process_mask(
    mask: np.ndarray, closing_radius: int = 2, min_size: int = None
) -> np.ndarray:
    """
    Process a binary mask to retain connected components based on size.

    Parameters
    ----------
    mask : numpy.ndarray, bool
        2D or 3D binary mask (boolean or 0/1 values).
    closing_radius : int, optional
        Radius for the structuring element used in binary closing.
        Default is 2.
    min_size : int, optional
        Minimum size (in pixels) for connected components to be retained.
        If None (default), only the largest connected component is retained.

    Returns
    -------
    final_mask : numpy.ndarray, bool
        A binary mask containing either the largest connected component
        or all connected components above the specified min_size
        (falling back to the largest component if none meet the min_size).
    """
    # Choose the appropriate structuring element based on image dimensions.
    if mask.ndim == 3:
        selem = ball(closing_radius)
    elif mask.ndim == 2:
        selem = disk(closing_radius)
    else:
        raise ValueError("Input volume must be 2D or 3D.")

    mask = binary_fill_holes(binary_closing(mask, selem))

    return size_filter(mask, min_size)


def size_filter(mask, min_size = None):
    # Label connected components in the filled mask.
    labeled_array, num_features = label(
        mask
    )
    _LOGGER.debug(f"Number of connected features: {num_features}")
    if num_features < 2**16:
        labeled_array = labeled_array.astype(np.uint16)

    if num_features == 0:
        raise ValueError("No connected components found in the image.")

    # Calculate sizes for each component (ignore background label 0).
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # ignore background
    
    if min_size is None:
        # Default: keep only the largest connected component.
        largest_component = np.argmax(component_sizes)
        final_mask = labeled_array == largest_component
    else:
        # Keep all components above the given size.
        valid_labels = np.where(component_sizes >= min_size)[0]
        if len(valid_labels) == 0:
            # Fall back to largest component if none meet min_size.
            _LOGGER.warning(
                "Minimum size component not satisfied, "
                "falling back to largest."
            )
            largest_component = np.argmax(component_sizes)
            final_mask = labeled_array == largest_component
        else:
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
