import os
import glob
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import zarr
from numcodecs import blosc
import tifffile
import dask
import dask.array as da
from distributed import Client, LocalCluster
from scipy.ndimage import gaussian_filter
from dask_image.ndfilters import gaussian_filter as gaussian_filter_dask

from exaspim_flatfield_correction.flatfield import (
    fit_basic,
    transform_basic,
    subtract_bkg,
)
from exaspim_flatfield_correction.splinefit import (
    percentile_project,
    get_correction_func,
)
from exaspim_flatfield_correction.background import estimate_bkg
from exaspim_flatfield_correction.utils.mask_utils import upscale_mask_nearest
from exaspim_flatfield_correction.utils.zarr_utils import (
    store_ome_zarr,
    parse_ome_zarr_transformations,
)
from exaspim_flatfield_correction.utils.metadata_utils import (
    create_processing_metadata,
)
from exaspim_flatfield_correction.utils.utils import (
    get_parent_s3_path,
    read_bkg_image,
    get_bkg_path,
    resize,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


def load_mask_from_dir(mask_dir: str, tile_name: str) -> np.ndarray:
    """
    Load a mask file for a given tile from a directory.

    Parameters
    ----------
    mask_dir : str
        Directory containing mask files.
    tile_name : str
        Name of the tile to match the mask file.

    Returns
    -------
    np.ndarray
        Loaded mask image as a numpy array.

    Raises
    ------
    Exception
        If no mask file is found for the given tile.
    """
    maskf = None
    for root, _, files in os.walk(mask_dir, followlinks=True):
        for f in files:
            if "_".join(tile_name.split("_")[:2]) in f:
                maskf = f
    if maskf is None:
        raise Exception(f"No mask file found for tile: {tile_name}")
    maskp = os.path.join(root, maskf)
    return tifffile.imread(maskp)


def parse_inputs(args: argparse.Namespace) -> dict:
    """
    Parse input arguments and metadata to determine tile paths and
    processing parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Dictionary containing tile paths, output path, method, resolution,
        and binned channel info.
    """
    if args.zarr != "":
        tile_paths = [args.zarr]
        out_path = args.output
        method = args.method
        res = str(args.res)
        binned_channel = None
    else:
        try:
            tile_file = glob.glob("../data/tile_*.json")[0]
        except IndexError:
            raise ValueError(
                "No tile metadata file found. "
                "Please provide a valid zarr file "
                "or a tile metadata JSON file."
            )
        with open(tile_file, "r") as f:
            meta = json.load(f)
        tile_paths = list(sorted(meta["tile_paths"]))
        out_path = meta["output_zarr"]
        method = meta["method"]
        binned_channel = meta["binned_channel"]
        res = str(meta["resolution"])
    params = {
        "tile_paths": tile_paths,
        "out_path": out_path,
        "method": method,
        "res": res,
        "binned_channel": binned_channel,
    }
    return params


def get_results_dir() -> str:
    """
    Get the results directory from the EXASPIM_RESULTS_DIR environment
    variable or use the default path.

    Returns
    -------
    str
        Absolute path to the results directory.
    """
    return os.environ.get(
        "EXASPIM_RESULTS_DIR",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../results")
        ),
    )


def background_subtraction(
    tile_path: str,
    full_res: da.Array,
    z: zarr.hierarchy.Group,
    is_binned_channel: bool = False,
) -> tuple[da.Array, np.ndarray]:
    """
    Perform background subtraction on the full resolution image.

    Parameters
    ----------
    tile_path : str
        Path to the tile zarr.
    full_res : dask.array.Array
        Full resolution image as a dask array.
    z : zarr.hierarchy.Group
        Opened zarr group for the tile.
    is_binned_channel : bool, optional
        Whether the tile is a binned channel, by default False.

    Returns
    -------
    tuple[dask.array.Array, np.ndarray]
        Tuple of background-subtracted full resolution dask array and the
        estimated background as a numpy array.
    """
    # Set to true to use the reference background image loaded from S3
    use_reference_bkg = False
    if use_reference_bkg:
        bkg_path = get_bkg_path(tile_path)
        bkg = read_bkg_image(bkg_path).astype(np.float32)
    else:
        bkg_res = "0" if is_binned_channel else "3"
        _LOGGER.info(f"Using resolution {bkg_res} for background estimation")
        low_res = da.from_zarr(z[bkg_res]).squeeze().astype(np.float32)
        bkg = estimate_bkg(gaussian_filter_dask(low_res, sigma=1).compute())

    full_res = subtract_bkg(
        full_res,
        da.from_array(
            resize(bkg, full_res.shape[1:]), chunks=full_res.chunksize[1:]
        ),
    )
    return full_res, bkg


def flatfield_reference(full_res: da.Array, flatfield_path: str) -> da.Array:
    """
    Apply reference flatfield correction to the image.

    Parameters
    ----------
    full_res : dask.array.Array
        Full resolution image as a dask array.
    flatfield_path : str
        Path to the flatfield image (local or s3).

    Returns
    -------
    dask.array.Array
        Flatfield-corrected image as a dask array.

    Raises
    ------
    ValueError
        If flatfield_path is not provided.
    """
    if flatfield_path is None:
        raise ValueError(
            "--flatfield-path must be provided when using the "
            "'reference' method."
        )
    if flatfield_path.startswith("s3://"):
        flatfield = read_bkg_image(flatfield_path).astype(np.float32)
    else:
        flatfield = tifffile.imread(flatfield_path).astype(np.float32)
    if flatfield.shape[-2:] != full_res.shape[-2:]:
        flatfield = resize(flatfield, full_res.shape[-2:])
    flatfield = flatfield / flatfield.max()
    flatfield = da.from_array(flatfield, chunks=(256, 256))
    corrected = da.clip(full_res / flatfield[np.newaxis], 0, 2**16 - 1)
    return corrected


def flatfield_basicpy(
    full_res: da.Array,
    z: zarr.hierarchy.Group,
    is_binned_channel: bool,
    bkg: np.ndarray = None,
) -> da.Array:
    """
    Apply basicpy-based flatfield correction to the image.

    Parameters
    ----------
    full_res : dask.array.Array
        Full resolution image as a dask array.
    z : zarr.hierarchy.Group
        Opened zarr group for the tile.
    is_binned_channel : bool
        Whether the tile is a binned channel.
    bkg : np.ndarray, optional
        Background image as a numpy array, by default None.

    Returns
    -------
    dask.array.Array
        Flatfield-corrected image as a dask array.
    """
    basicpy_res = "1" if is_binned_channel else "4"
    low_res = da.from_zarr(z[basicpy_res]).squeeze()
    if bkg is not None:
        low_res = subtract_bkg(
            low_res,
            da.from_array(
                resize(bkg, low_res.shape[1:]), chunks=low_res.chunksize[1:]
            ),
        )
    fit = fit_basic(
        low_res.compute(),
        autotune=False,
        get_darkfield=False,
        sort_intensity=True,
        shuffle_frames=False,
    )
    corrected = transform_basic(full_res, fit)
    return corrected


def _preprocess_mask(mask_dir: str, tile_name: str, low_res_shape) -> da.Array:
    """
    Load, upscale, and save mask as zarr, then reload as dask array.
    """
    mask_name = str(Path(get_results_dir()) / f"{tile_name}_mask.zarr")
    mask = load_mask_from_dir(mask_dir, tile_name)
    if mask.shape != low_res_shape:
        mask = (
            upscale_mask_nearest(
                da.from_array(mask, chunks=(128, 256, 256)),
                low_res_shape,
                chunks=(128, 256, 256),
            )
            .astype(np.uint8)
            .compute()
        )
    zarr.save_array(
        str(mask_name),
        mask,
        chunks=(128, 256, 256),
        compressor=blosc.Blosc(cname="zstd", clevel=1),
    )
    return da.from_zarr(mask_name, chunks=(128, 256, 256))


def flatfield_fitting(
    full_res: da.Array,
    z: zarr.hierarchy.Group,
    is_binned_channel: bool,
    mask_dir: str,
    tile_name: str,
    out_mask_path: str,
    coordinate_transformations: dict,
    overwrite: bool,
    n_levels: int,
    config: dict,
) -> da.Array:
    """
    Apply fitting-based flatfield correction to the image using a mask
    and configurable parameters.

    Parameters
    ----------
    full_res : dask.array.Array
        Full resolution image as a dask array.
    z : zarr.hierarchy.Group
        Opened zarr group for the tile.
    is_binned_channel : bool
        Whether the tile is a binned channel.
    mask_dir : str
        Directory containing mask files.
    tile_name : str
        Name of the tile being processed.
    out_mask_path : str
        Output path for the mask zarr.
    coordinate_transformations : dict
        Dictionary of coordinate transformation parameters.
    overwrite : bool
        Whether to overwrite existing outputs.
    n_levels : int
        Number of zarr pyramid levels.
    config : dict
        Dictionary of fitting parameters (see get_fitting_config).

    Returns
    -------
    dask.array.Array
        Flatfield-corrected image as a dask array.
    """
    fitting_res = "0" if is_binned_channel else "3"
    low_res = da.from_zarr(z[fitting_res]).squeeze().astype(np.float32)

    mask = _preprocess_mask(mask_dir, tile_name, low_res.shape)
    mask_2d_xy = mask.max(axis=0).compute()
    mask_2d_yz = mask.max(axis=2).compute()

    med_factor = (
        config.get("med_factor_binned", 2)
        if is_binned_channel
        else config.get("med_factor_unbinned", 5)
    )
    nan_med = np.nanmedian(da.where(mask, low_res, np.nan).compute())
    low_res_clipped = np.clip(low_res, 0, nan_med * med_factor)

    del low_res, nan_med

    percentile = config.get("percentile", 99)
    xy_proj = percentile_project(
        low_res_clipped, axis=0, percentile=percentile
    )
    xy_proj = gaussian_filter(xy_proj, sigma=config.get("gaussian_sigma", 2))
    yz_proj = percentile_project(
        low_res_clipped, axis=2, percentile=percentile
    )
    yz_proj = gaussian_filter(yz_proj, sigma=config.get("gaussian_sigma", 2))

    del low_res_clipped

    mask_upscaled = upscale_mask_nearest(
        mask, full_res.shape, chunks=(128, 256, 256)
    ).astype(np.uint8)
    mask_path = out_mask_path.rstrip("/") + f"/{tile_name}"
    store_ome_zarr(
        mask_upscaled,
        mask_path,
        n_levels,
        coordinate_transformations["scale"][-3:],
        coordinate_transformations["translation"],
        overwrite=overwrite,
    )
    mask_upscaled = da.from_zarr(mask_path, "0").squeeze()

    fit_x, median_xy = get_correction_func(
        xy_proj,
        mask_2d_xy,
        axis=0,
        new_width=full_res.shape[2],
        spline_smoothing=config.get("spline_smoothing", 0.01),
        limits=config.get("limits_xy", (0.25, 1.2)),
    )
    if median_xy == 0:
        corrected = full_res
    else:
        correction_x = fit_x.reshape(1, 1, -1)
        corrected = da.where(mask_upscaled, full_res / correction_x, full_res)

        fit_z, median_yz = get_correction_func(
            yz_proj,
            mask_2d_yz,
            axis=1,
            new_width=full_res.shape[0],
            spline_smoothing=config.get("spline_smoothing", 0.01),
            limits=config.get("limits_z", (0.25, 1.2)),
        )
        correction_z = fit_z.reshape(-1, 1, 1)
        corrected = da.where(
            mask_upscaled, corrected / correction_z, corrected
        )

        _LOGGER.info(f"median xy: {median_xy}")
        global_factor = (
            config.get("global_factor_binned", 9000)
            if is_binned_channel
            else config.get("global_factor_unbinned", 70)
        )
        corrected = da.where(
            mask_upscaled, corrected * (global_factor / median_xy), corrected
        )
        corrected = da.clip(corrected, 0, 2**16 - 1)

    return corrected


def save_metadata(
    data_process, out_path: str, tile_name: str, tile_path: str
) -> None:
    """
    Save process and metadata paths JSON for a tile.

    Parameters
    ----------
    data_process : Any
        Processing metadata object with a model_dump_json() method.
    out_path : str
        Output zarr path for the tile.
    tile_name : str
        Name of the tile being processed.
    tile_path : str
        Path to the input tile zarr.
    """
    process_json = data_process.model_dump_json()
    process_json_path = str(
        Path(get_results_dir())
        / f"process_{Path(out_path).parent.name}_{tile_name}.json"
    )
    with open(process_json_path, "w") as f:
        f.write(process_json)

    input_metadata_path = get_parent_s3_path(get_parent_s3_path(tile_path))
    output_metadata_path = get_parent_s3_path(out_path)
    metadata_json_path = str(
        Path(get_results_dir())
        / f"metadata_paths_{Path(out_path).parent.name}_{tile_name}.json"
    )
    with open(metadata_json_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "input_metadata": input_metadata_path,
                    "output_metadata": output_metadata_path,
                }
            )
        )


def get_fitting_config() -> dict:
    """
    Return the default config dictionary for fitting correction.

    Returns
    -------
    dict
        Dictionary of default fitting parameters for flatfield_fitting().
    """
    return {
        "med_factor_binned": 2,
        "med_factor_unbinned": 5,
        "percentile": 99,
        "gaussian_sigma": 2,
        "spline_smoothing": 0.01,
        "limits_xy": (0.25, 1.2),
        "limits_z": (0.25, 1.2),
        "global_factor_binned": 9000,
        "global_factor_unbinned": 70,
    }


def get_channel_resolution(
    tile_name: str, binned_channel: str, binned_res: str, res: str
) -> tuple[bool, str]:
    """
    Determine if the tile is a binned channel and select the appropriate
    zarr resolution.

    Parameters
    ----------
    tile_name : str
        Name of the tile being processed.
    binned_channel : str
        Name of the binned channel pattern.
    binned_res : str
        Resolution to use for binned channels.
    res : str
        Resolution to use for unbinned channels.

    Returns
    -------
    tuple[bool, str]
        Tuple containing a boolean indicating if the tile is a binned channel,
        and the selected resolution.
    """
    is_binned = binned_channel is not None and binned_channel in tile_name
    return is_binned, binned_res if is_binned else res


def set_dask_config():
    """
    Set Dask configuration for memory management and temporary directory.
    """
    dask_tmp_dir = os.path.join(get_results_dir(), "dask-tmp")
    dask.config.set(
        {
            "temporary-directory": dask_tmp_dir,
            "distributed.worker.memory.target": 0.7,
            "distributed.worker.memory.spill": 0.8,
            "distributed.worker.memory.pause": 0.9,
            "distributed.worker.memory.terminate": 0.95,
            "distributed.scheduler.allowed-failures": 100,
        }
    )


def parse_and_validate_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for the flatfield
    correction pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed and validated command-line arguments.

    Raises
    ------
    ValueError
        If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zarr",
        type=str,
        required=True,
        help="Input zarr path for the tile data",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output zarr path for the corrected data",
    )
    parser.add_argument(
        "--res",
        type=str,
        default="0",
        help="Resolution level to process (default: 0)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["basicpy", "reference", "fitting"],
        default="fitting",
    )
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument(
        "--skip-flat-field", action="store_true", default=False
    )
    parser.add_argument("--skip-bkg-sub", action="store_true", default=False)
    parser.add_argument("--flatfield-path", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-level", type=str, default=logging.INFO)
    parser.add_argument(
        "--n-levels",
        type=int,
        default=1,
        help="Number of zarr pyramid levels (default: 1)",
    )
    args = parser.parse_args()

    if args.method == "fitting" and args.mask_dir is None:
        raise ValueError(
            "Mask directory (--mask-dir) must be specified when using the "
            "'fitting' method."
        )
    if args.skip_bkg_sub and args.skip_flat_field:
        raise ValueError(
            "Cannot skip both flat field correction and "
            "background subtraction. At least one must be performed."
        )
    return args


def main() -> None:
    """
    Main entry point for the flatfield correction pipeline.
    Parses arguments, sets up Dask, processes each tile,
    and saves results and metadata.
    """
    args = parse_and_validate_args()
    _LOGGER.setLevel(args.log_level)

    _LOGGER.info(f"args: {args}")

    params = parse_inputs(args)
    _LOGGER.info(f"Parsed parameters: {params}")

    tile_paths = params["tile_paths"]
    out_path = params["out_path"]
    method = params["method"]
    res = params["res"]
    binned_channel = params["binned_channel"]

    set_dask_config()

    start_date_time = datetime.now()

    client = Client(
        LocalCluster(
            processes=False, n_workers=1, threads_per_worker=args.num_workers
        )
    )
    _LOGGER.info(f"Dask client: {client}")

    out_mask_path = out_path.replace("/SPIM.ome.zarr", "/mask/SPIM.ome.zarr")
    data_process = create_processing_metadata(
        args, tile_paths[0], out_path, start_date_time, res
    )
    binned_res = "0"

    for tile_path in tile_paths:
        tile_name = Path(tile_path).name
        _LOGGER.info(f"Processing tile: {tile_name}")
        try:
            is_binned_channel, resolution = get_channel_resolution(
                tile_name, binned_channel, binned_res, res
            )
            _LOGGER.info(f"{tile_name} is binned: {is_binned_channel}")

            z = zarr.open(tile_path, mode="r")
            coordinate_transformations = parse_ome_zarr_transformations(
                z, resolution
            )
            _LOGGER.info(
                f"Coordinate transformations: {coordinate_transformations}"
            )

            full_res = da.from_zarr(z[resolution]).squeeze().astype(np.float32)
            _LOGGER.info(f"Full resolution array shape: {full_res.shape}")

            bkg = None
            if not args.skip_bkg_sub:
                _LOGGER.info("Performing background subtraction")
                full_res, bkg = background_subtraction(
                    tile_path, full_res, z, is_binned_channel
                )

            if not args.skip_flat_field:
                if method == "reference":
                    corrected = flatfield_reference(
                        full_res, args.flatfield_path
                    )
                elif method == "basicpy":
                    corrected = flatfield_basicpy(
                        full_res, z, is_binned_channel, bkg
                    )
                elif method == "fitting":
                    fitting_config = get_fitting_config()
                    corrected = flatfield_fitting(
                        full_res,
                        z,
                        is_binned_channel,
                        args.mask_dir,
                        tile_name,
                        out_mask_path,
                        coordinate_transformations,
                        args.overwrite,
                        args.n_levels,
                        fitting_config,
                    )
                else:
                    _LOGGER.error(f"Invalid method: {method}")
                    raise ValueError(f"Invalid method: {method}")
            else:
                corrected = full_res

            corrected = corrected.astype(np.uint16)
            _LOGGER.info(f"Corrected array dtype: {corrected.dtype}")

            t0 = time.time()
            store_ome_zarr(
                corrected,
                out_path.rstrip("/") + f"/{tile_name}",
                args.n_levels,
                coordinate_transformations["scale"][-3:],
                coordinate_transformations["translation"],
                overwrite=args.overwrite,
            )
            _LOGGER.info(f"Storing OME-Zarr took {time.time() - t0:.2f}s")

            data_process.end_date_time = datetime.now()
            save_metadata(data_process, out_path, tile_name, tile_path)
        except Exception as e:
            _LOGGER.error(
                f"Error processing tile {tile_name}: {e}", exc_info=True
            )
            raise


if __name__ == "__main__":
    main()
