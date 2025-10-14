import io
import logging
import os
import re
import math
import subprocess
from pathlib import Path

import boto3
import dask.array as da
import matplotlib
import numpy as np
import tifffile
from botocore.exceptions import ClientError
from skimage.transform import resize as _resize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LOGGER = logging.getLogger(__name__)


def compose_image(
    img: list[da.Array], rows: int = 3, cols: int = 5
) -> da.Array:
    """
    Compose a grid image from a list of 3D Dask arrays (C, Y, X).

    Parameters
    ----------
    img : list of dask.array.Array
        List of 3D Dask arrays to compose (C, Y, X). All arrays must have
        the same shape.
    rows : int, optional
        Number of rows in the composed image grid. Default is 3.
    cols : int, optional
        Number of columns in the composed image grid. Default is 5.

    Returns
    -------
    dask.array.Array
        Composed image as a Dask array.
    """
    img_composed = da.empty(
        (
            img[0].shape[0],
            img[0].shape[1] * rows,
            img[0].shape[2] * cols,
        ),
        chunks=img[0].chunksize[-3:],
        dtype=np.uint16,
    )

    i = 0
    y = 0
    x = img[0].shape[2] * (cols - 1)
    for c in range(cols - 1, -1, -1):
        x = img[0].shape[2] * c
        for r in range(rows):
            y = img[0].shape[1] * r
            img_composed[
                :, y : y + img[0].shape[1], x : x + img[0].shape[2]
            ] = img[i]
            i += 1
    return img_composed


def get_parent_s3_path(s3_url: str) -> str:
    """
    Extract the parent path of an S3 object.

    Parameters
    ----------
    s3_url : str
        The S3 URL in the format 's3://bucket-name/path/to/object'.

    Returns
    -------
    str
        The parent S3 path in the format 's3://bucket-name/path/to', or the
        bucket root if no parent exists.

    Raises
    ------
    ValueError
        If the provided URL is not a valid S3 URL or does not have a parent.
    """
    if not s3_url.startswith("s3://"):
        raise ValueError("Provided URL is not a valid S3 URL.")

    # Remove 's3://' and split into bucket name and object path
    parts = s3_url[5:].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("S3 URL does not contain a valid object path.")

    bucket_name, object_path = parts

    # Remove trailing slash if present
    object_path = object_path.rstrip("/")

    # Check if there's a parent path to extract
    if "/" not in object_path:
        # No parent path available
        return f"s3://{bucket_name}"

    # Extract parent path
    parent_path = "/".join(object_path.rsplit("/", 1)[:-1])

    return f"s3://{bucket_name}/{parent_path}"


def check_s3_path_exists(s3_url: str) -> bool:
    """
    Check if a path exists on S3.

    Parameters
    ----------
    s3_url : str
        S3 URL in the format 's3://bucket/key'.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
    """
    s3 = boto3.client("s3")
    if not s3_url.startswith("s3://") or "/" not in s3_url[5:]:
        return False
    s3_url_parts = s3_url.replace("s3://", "").split("/", 1)
    if len(s3_url_parts) != 2:
        return False
    bucket = s3_url_parts[0]
    path = s3_url_parts[1]
    try:
        s3.head_object(Bucket=bucket, Key=path)
        return True
    except ClientError:
        return False


def get_abs_path(relative_path: str) -> str:
    """
    Get an absolute path from a path relative to the script location.

    Parameters
    ----------
    relative_path : str
        Path relative to the project root or script location.

    Returns
    -------
    str
        Absolute path as a string.
    """
    return str((Path(__file__).parent.parent.parent / relative_path).resolve())


def read_bkg_image(s3_url: str) -> np.ndarray:
    """
    Read a TIFF image from an S3 URL and return as a numpy array.

    Parameters
    ----------
    s3_url : str
        S3 URL to the TIFF image.

    Returns
    -------
    np.ndarray
        Image loaded as a numpy array.

    Raises
    ------
    ValueError
        If the S3 URL is malformed.
    """
    # Parse the S3 URL
    if not s3_url.startswith("s3://") or "/" not in s3_url[5:]:
        raise ValueError(f"Malformed S3 URL: {s3_url}")
    s3_url_parts = s3_url.replace("s3://", "").split("/", 1)
    if len(s3_url_parts) != 2:
        raise ValueError(f"Malformed S3 URL: {s3_url}")
    bucket = s3_url_parts[0]
    path = s3_url_parts[1]
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=path)
    tiff_file_content = response["Body"].read()
    numpy_array = tifffile.imread(io.BytesIO(tiff_file_content))
    return numpy_array


def upload_artifacts(results_dir: str, destination: str) -> None:
    """Upload local artifacts to an S3 destination using the AWS CLI.

    Parameters
    ----------
    results_dir : str
        Local directory containing the artifacts to upload.
    destination : str
        Target S3 URI where the artifacts should be copied.

    Raises
    ------
    subprocess.CalledProcessError
        If the AWS CLI copy command fails.
    """

    _LOGGER.info("Uploading artifacts from %s to %s", results_dir, destination)
    try:
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                "--recursive",
                results_dir,
                destination,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        _LOGGER.error(
            "Failed to upload artifacts to %s", destination, exc_info=True
        )
        raise


def get_bkg_path(raw_tile_path: str) -> str:
    """
    Get the S3 path to the background image for a given raw tile path.

    Parameters
    ----------
    raw_tile_path : str
        Path to the raw tile (S3 or local).

    Returns
    -------
    str
        S3 path to the background image.

    Raises
    ------
    FileNotFoundError
        If no background image is found for the tile.
    """
    regex1 = re.compile(r"tile_x_\d+_y_\d+_z_\d+_ch_\d+")
    regex2 = re.compile(r"tile_x_\d+_y_\d+_z_\d+")
    regex3 = re.compile(r"tile_\d+_ch_\d+")

    p = Path(raw_tile_path).stem

    m = regex1.match(p)
    if m:
        match = m.group(0)
        bkg_tile = f"bkg_{match}.tiff"
        bkg_path = (
            get_parent_s3_path(get_parent_s3_path(raw_tile_path))
            + "/derivatives/"
            + bkg_tile
        )
        if check_s3_path_exists(bkg_path):
            return bkg_path

    m = regex2.match(p)
    if m:
        match = m.group(0)
        bkg_tile = f"bkg_{match}.tiff"
        bkg_path = (
            get_parent_s3_path(get_parent_s3_path(raw_tile_path))
            + "/derivatives/"
            + bkg_tile
        )
        if check_s3_path_exists(bkg_path):
            return bkg_path

    m = regex3.match(p)
    if m:
        match = m.group(0)
        bkg_tile = f"{match}_background_collection.tiff"
        bkg_path = (
            get_parent_s3_path(get_parent_s3_path(raw_tile_path))
            + "/derivatives/"
            + bkg_tile
        )
        if check_s3_path_exists(bkg_path):
            return bkg_path

    raise FileNotFoundError(
        f"Could not find background image for tile {raw_tile_path}"
    )


def resize(
    im: np.ndarray, shape: tuple[int, ...], order: int = 3
) -> np.ndarray:
    """
    Resize a numpy array to a new shape using skimage.transform.resize.

    Parameters
    ----------
    im : np.ndarray
        Input image array.
    shape : tuple of int
        Desired output shape.
    order : int, optional
        Interpolation order (0=nearest, 1=linear, 3=bicubic, etc).
        Default is 3.

    Returns
    -------
    np.ndarray
        Resized image array.
    """
    return _resize(
        im,
        shape,
        anti_aliasing=False,
        mode="reflect",
        clip=True,
        order=order,  # bicubic interpolation
        preserve_range=True,
    )


def resize_dask(
    image: da.Array,
    scale_factor: "float | tuple[float, float, float]",
    order: int = 1,
    output_chunks: tuple[int, int, int] = (128, 256, 256),
) -> da.Array:
    """
    Resize a 3D Dask array using an affine transformation.

    Parameters
    ----------
    image : dask.array.Array
        The input 3D Dask array to be resized.
    scale_factor : float
        The scaling factor for each axis. For example, 2.0 will double the
        size.
    order : int, optional
        The order of the interpolation. Use order=0 for nearest-neighbor
        (good for binarymasks), or higher orders for smoother results.
        Default is 1 (linear interpolation).
    output_chunks : tuple, optional
        The desired chunk size for the output Dask array.
        Default is (128, 256, 256).

    Returns
    -------
    dask.array.Array
        The resized Dask array.
    """
    from dask_image.ndinterp import affine_transform

    # Determine per-axis scaling factors
    if isinstance(scale_factor, (tuple, list)):
        sz, sy, sx = map(float, scale_factor)
    else:
        sz = sy = sx = float(scale_factor)

    # Construct a 4x4 homogeneous affine transformation matrix.
    # The matrix maps output coordinates into input coordinates.
    # Scaling factors are inverted because of this coordinate mapping.
    matrix = np.array(
        [
            [1 / sz, 0, 0, 0],
            [0, 1 / sy, 0, 0],
            [0, 0, 1 / sx, 0],
            [0, 0, 0, 1],
        ]
    )

    # Calculate the new output shape (assumes image has at least 3 dimensions).
    new_shape = (
        int(image.shape[0] * sz),
        int(image.shape[1] * sy),
        int(image.shape[2] * sx),
    )

    # Apply the affine transformation.
    resized_image = affine_transform(
        image,
        matrix=matrix,
        order=order,
        output_shape=new_shape,
        output_chunks=output_chunks,
    )

    return resized_image


def save_correction_curve_plot(
    curve: "np.ndarray | list[float]",
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: str,
    dpi: int = 150,
) -> None:
    """
    Save a simple line plot for a 1D correction curve to a PNG file.

    Parameters
    ----------
    curve : array-like
        1D array of correction factors to plot.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    out_png : str
        Output PNG path.
    dpi : int, optional
        Figure resolution, default 150.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(curve)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def array_chunks(arr: da.Array) -> tuple[int, ...]:
    """Return the first chunk length from each axis of a Dask array."""

    return tuple(int(axis_chunks[0]) for axis_chunks in arr.chunks)


def chunks_2d(arr: da.Array) -> tuple[int, ...]:
    """Return chunk lengths for the last two axes of a Dask array."""

    if arr.ndim < 2:
        return (int(arr.chunks[-1][0]),)
    return tuple(int(axis_chunks[0]) for axis_chunks in arr.chunks[-2:])


def load_mask_from_dir(mask_dir: str, tile_name: str) -> np.ndarray:
    """Load the binary mask that corresponds to ``tile_name``.

    Parameters
    ----------
    mask_dir : str
        Directory that stores candidate mask files.
    tile_name : str
        Tile identifier used to locate the matching mask.

    Returns
    -------
    numpy.ndarray
        Mask stored on disk with shape compatible with the tile.

    Raises
    ------
    ValueError
        If the directory or tile name is invalid.
    FileNotFoundError
        If no file matching the tile can be located.
    """
    if tile_name is None or tile_name == "":
        raise ValueError("Tile name must be provided to load the mask.")
    if mask_dir is None or not os.path.isdir(mask_dir):
        raise ValueError(
            f"Mask directory {mask_dir} does not exist or is not a directory."
        )
    _LOGGER.info(
        f"Loading mask from directory: {mask_dir} for tile: {tile_name}"
    )
    tile_prefix = "_".join(tile_name.split("_")[:2])
    for root, _, files in os.walk(mask_dir, followlinks=True):
        for f in files:
            if tile_prefix in f:
                maskp = os.path.join(root, f)
                _LOGGER.info(f"Found mask file: {maskp}")
                return tifffile.imread(maskp)
    raise FileNotFoundError(f"No mask file found for tile: {tile_name}")


def extract_channel_from_tile_name(tile_name: str) -> str | None:
    """Extract a numeric channel identifier from a tile name.

    Parameters
    ----------
    tile_name : str
        File or directory name describing the tile (e.g., ``tile_000017_ch_488``).

    Returns
    -------
    str or None
        The extracted channel digits, or ``None`` when no pattern is found.
    """

    match = re.search(r"_ch_(\d+)", tile_name)
    if match:
        return match.group(1)

    match = re.search(r"_ch(\d+)", tile_name)
    if match:
        return match.group(1)

    return None


def weighted_percentile(
    data: da.Array,
    mask: da.Array,
    percentile: float,
    weights: da.Array | None = None,
) -> float:
    """
    Compute a weighted percentile from a masked volume using a histogram.

    Parameters
    ----------
    data : dask.array.Array
        Input volume whose values are assumed to be within the 16-bit range.
    mask : dask.array.Array
        Boolean or integer mask identifying voxels to include in the percentile
        computation. Non-zero entries are treated as foreground.
    percentile : float
        Desired percentile in ``[0, 100]``.
    weights : dask.array.Array or None, optional
        Per-voxel weights aligned with ``data``. When provided, non-zero mask
        elements are weighted by these values using the inverted CDF rule.

    Returns
    -------
    float
        The weighted percentile value.

    Raises
    ------
    ValueError
        If ``percentile`` lies outside ``[0, 100]``, the histogram encounter
        non-finite values, or no voxels are selected by the mask.
    """

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Percentile must be between 0 and 100, received {percentile}")

    use_inverted = weights is not None

    if weights is not None:
        hist_weights = (weights * mask).astype(np.float32)
    else:
        hist_weights = mask.astype(np.float32)

    data_min = int(data.min().compute())
    data_max = int(data.max().compute())
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        raise ValueError("Encountered non-finite values while computing percentile range.")
    if data_min == data_max:
        return float(data_min)

    bins = int(math.ceil(data_max) - math.floor(data_min)) + 1
    hist_range = (float(data_min), float(data_max + 1))

    hist_weights = hist_weights.rechunk(data.chunks)

    counts, edges = da.histogram(
        data,
        bins=bins,
        range=hist_range,
        weights=hist_weights,
    )
    counts_np = counts.compute()
    total_weight = counts_np.sum()
    if total_weight == 0:
        raise ValueError("No voxels selected by mask; cannot compute percentile.")

    edges_np = np.asarray(edges)
    target_weight = (percentile / 100.0) * total_weight
    cdf = np.cumsum(counts_np)
    bin_idx = int(np.searchsorted(cdf, target_weight, side="left"))
    bin_idx = min(bin_idx, len(counts_np) - 1)

    lower_edge = edges_np[bin_idx]
    upper_edge = edges_np[bin_idx + 1]
    prev_cumulative = cdf[bin_idx - 1] if bin_idx > 0 else 0.0
    bin_weight = counts_np[bin_idx]
    in_bin_target = np.clip(target_weight - prev_cumulative, 0.0, bin_weight)

    if bin_weight == 0 or lower_edge == upper_edge:
        return float(lower_edge)

    if use_inverted:
        return float(lower_edge)

    fraction = in_bin_target / bin_weight
    return float(lower_edge + fraction * (upper_edge - lower_edge))
