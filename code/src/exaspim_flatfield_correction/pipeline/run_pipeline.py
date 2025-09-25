import os
import glob
import time
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import zarr
from numcodecs import blosc
blosc.use_threads = False
import tifffile
import dask
import dask.array as da
from dask.distributed import performance_report
from distributed import Client, LocalCluster
from scipy.ndimage import gaussian_filter
from dask_image.ndfilters import gaussian_filter as gaussian_filter_dask

from exaspim_flatfield_correction.flatfield import (
    fit_basic,
    transform_basic,
    subtract_bkg,
)
from exaspim_flatfield_correction.splinefit import (
    masked_axis_profile,
    percentile_project,
    rescale_spline,
)
from exaspim_flatfield_correction.background import estimate_bkg
from exaspim_flatfield_correction.utils.mask_utils import (
    gmm_probability_mask_features,
    gmm_probability_mask,
    project_probability_mask,
    size_filter,
    upscale_mask_nearest,
    get_mask
)
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
    save_correction_curve_plot,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


@dataclass
class MaskArtifacts:
    probability_volume: da.Array
    mask_low_res: da.Array


def get_mem_limit() -> int | str:
    """
    Return a value suitable for Dask’s LocalCluster(memory_limit=…).

    • If the CO_MEMORY environment variable is set, treat it as **bytes** and
      return it as an int.
    • If CO_MEMORY is unset (or empty), return the string 'auto'.

    Raises
    ------
    ValueError
        If CO_MEMORY is set to a non-integer value.
    """
    raw = os.getenv("CO_MEMORY")
    if not raw:
        return "auto"

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"CO_MEMORY must be an integer number of bytes; got {raw!r}"
        ) from exc


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
    if tile_name is None or tile_name == "":
        raise ValueError("Tile name must be provided to load the mask.")
    if mask_dir is None or not os.path.isdir(mask_dir):
        raise ValueError(f"Mask directory {mask_dir} does not exist or is not a directory.")
    _LOGGER.info(f"Loading mask from directory: {mask_dir} for tile: {tile_name}")
    tile_prefix = "_".join(tile_name.split("_")[:2])
    for root, _, files in os.walk(mask_dir, followlinks=True):
        for f in files:
            if tile_prefix in f:
                maskp = os.path.join(root, f)
                _LOGGER.info(f"Found mask file: {maskp}")
                return tifffile.imread(maskp)
    raise Exception(f"No mask file found for tile: {tile_name}")


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


def background_subtraction(
    tile_path: str,
    full_res: da.Array,
    z: zarr.hierarchy.Group,
    is_binned_channel: bool = False,
    use_reference_bkg: bool = False,
) -> tuple[da.Array, np.ndarray, np.ndarray | None]:
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
    use_reference_bkg : bool, optional
        If True, loads the reference background image from S3. If False, estimate background from data. Default is False.

    Returns
    -------
    tuple[dask.array.Array, np.ndarray, np.ndarray | None]
        Tuple containing the background-subtracted full resolution array, the
        estimated 2D background image, and the stack of background-dominated
        slices used for probability modelling (or None when unavailable).
    """
    if use_reference_bkg:
        bkg_path = get_bkg_path(tile_path)
        bkg = read_bkg_image(bkg_path).astype(np.float32)
        bkg_slices = None
    else:
        bkg_res = "0" if is_binned_channel else "3"
        _LOGGER.info(f"Using resolution {bkg_res} for background estimation")
        low_res = da.from_zarr(z[bkg_res]).squeeze().astype(np.float32)
        bkg, bkg_slices = estimate_bkg(gaussian_filter_dask(low_res, sigma=1).compute())

    full_res = subtract_bkg(
        full_res,
        da.from_array(
            resize(bkg, full_res.shape[1:]), chunks=full_res.chunksize[1:]
        ),
    )
    return full_res, bkg, bkg_slices


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
    tuple
        Tuple containing the corrected image, GMM probability volume,
        binary projection masks, percentile projections, fitted
        correction curves, and the mask artifacts for reuse.

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
    mask_dir: str = None,
    tile_name: str = None,
    max_slices: int = 100,
    working_size: int = 512,
    sort_intensity: bool = True,
    shuffle_frames: bool = False,
    autotune: bool = False,
    results_dir: str = None,
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
    mask_dir : str, optional
        Directory containing mask files, by default None.
    tile_name : str, optional
        Name of the tile being processed, by default None.
    max_slices : int, optional
        Maximum number of slices for basicpy, by default 100.
    working_size : int, optional
        Working size for basicpy, by default 512.
    sort_intensity : bool, optional
        Whether to sort by intensity, by default True.
    shuffle_frames : bool, optional
        Whether to shuffle frames, by default False.
    autotune : bool, optional
        Whether to autotune basicpy, by default False.
    results_dir : str, optional
        Directory to store intermediate mask zarr, by default None.

    Returns
    -------
    dask.array.Array
        Corrected volume after applying the BasicPy flatfield fit.
    """
    basicpy_res = "0" if is_binned_channel else "3"
    low_res = da.from_zarr(z[basicpy_res]).squeeze().astype(np.float32)
    if bkg is not None:
        low_res = subtract_bkg(
            low_res,
            da.from_array(
                resize(bkg, low_res.shape[1:]), chunks=low_res.chunksize[1:]
            ),
        )
    mask = None
    if mask_dir is not None and results_dir is not None:
        mask = _preprocess_mask(
            load_mask_from_dir(mask_dir, tile_name),
            low_res.shape,
            results_dir,
            tile_name
        ).compute()
    fit = fit_basic(
        low_res.compute(),
        autotune=autotune,
        get_darkfield=False,
        sort_intensity=sort_intensity,
        shuffle_frames=shuffle_frames,
        mask=mask,
        max_slices=max_slices,
        working_size=working_size,
    )
    corrected = transform_basic(full_res, fit)
    return corrected


def _preprocess_mask(mask: np.ndarray, low_res_shape: tuple, results_dir: str, tile_name: str) -> da.Array:
    """
    Upscale and save mask as zarr, then reload as dask array.
    """
    mask_name = str(Path(results_dir) / f"{tile_name}_mask.zarr")
    if mask.shape != low_res_shape:
        mask = upscale_mask_nearest(
                da.from_array(mask, chunks=(128, 256, 256)),
                low_res_shape,
                chunks=(128, 256, 256),
        ).compute()
        # keep only the largest connected component
        mask = size_filter(mask, k_largest=1, min_size=None)
    mask = mask.astype(np.uint8)
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
    out_probability_path: str,
    coordinate_transformations: dict,
    overwrite: bool,
    n_levels: int,
    config: dict,
    results_dir: str = None,
    bkg_slices: np.ndarray | None = None,
    mask_artifacts: MaskArtifacts | None = None,
) -> tuple:
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
    out_probability_path : str
        Output path for the probability volume OME-Zarr.
    coordinate_transformations : dict
        Dictionary of coordinate transformation parameters.
    overwrite : bool
        Whether to overwrite existing outputs.
    n_levels : int
        Number of zarr pyramid levels.
    config : dict
        Dictionary of fitting parameters (see get_fitting_config).
    results_dir : str, optional
        Directory to store intermediate mask zarr, by default None.
    bkg_slices : numpy.ndarray or None, optional
        Background-dominated slices from background estimation used to fit
        the background GMM. When None, probabilities revert to the
        foreground-only model.
    mask_artifacts : MaskArtifacts or None, optional
        Precomputed mask artifacts to reuse across tiles. When None, the
        artifacts are created using the current tile and returned for reuse.

    Returns
    -------
    tuple
        Tuple containing the corrected image, GMM probability volume,
        binary projection masks, percentile projections, and fitted
        correction curves.
    """
    fitting_res = "0" if is_binned_channel else "3"
    low_res = da.from_zarr(z[fitting_res]).squeeze().astype(np.float32)

    if mask_artifacts is None:
        _LOGGER.info("Creating mask artifacts using tile %s", tile_name)
        initial_mask = _preprocess_mask(
            load_mask_from_dir(mask_dir, tile_name),
            low_res.shape,
            results_dir,
            tile_name,
        ).astype(bool)

        gmm_n_components = config.get("gmm_n_components")
        gmm_max_samples = config.get("gmm_max_samples")
        gmm_batch_size = config.get("gmm_batch_size")
        gmm_random_state = config.get("gmm_random_state")

        _LOGGER.info(
            "Fitting GMM probabilities with n_components=%d, max_samples=%d, batch_size=%d",
            gmm_n_components,
            gmm_max_samples,
            gmm_batch_size,
        )
        if bkg_slices is None:
            raise ValueError(
                "Background slices are required to fit the mask GMM on the reference tile"
            )
        probability_volume = gmm_probability_mask_features(
            gaussian_filter_dask(low_res.astype(np.float32), sigma=1),
            initial_mask,
            da.from_array(bkg_slices, chunks=(128, 128, 128)),
            n_components_fg=4,
            n_components_bg=2,
            max_samples_fg=gmm_max_samples,
            max_samples_bg=gmm_max_samples,
            random_state=gmm_random_state,
            erosion_radius=10,
        ).astype(np.float32).persist()
        
        # Persist the soft tissue probabilities as an OME-Zarr dataset at the
        # resolution used for GMM fitting to retain spatial metadata.
        probability_path = out_probability_path.rstrip("/") + f"/{tile_name}"
        probability_transformations = parse_ome_zarr_transformations(z, fitting_res)
        probability_scale = probability_transformations["scale"]
        probability_translation = probability_transformations["translation"]

        store_ome_zarr(
            probability_volume,
            probability_path,
            4,
            tuple(probability_scale[-3:]),
            tuple(probability_translation),
            overwrite=overwrite,
            write_empty_chunks=False,
        )

        prob_threshold = config.get("mask_probability_threshold")
        prob_min_size = config.get("mask_probability_min_size")
        _LOGGER.info(
            "Projecting probability mask with threshold %.3f and min_size %s",
            prob_threshold,
            prob_min_size,
        )
        mask = get_mask(
            probability_volume,
            threshold=prob_threshold,
            min_size=prob_min_size,
        )
        mask = _preprocess_mask(mask, low_res.shape, results_dir, tile_name).astype(bool).persist()

        mask_artifacts = MaskArtifacts(
            probability_volume=probability_volume,
            mask_low_res=mask,
        )
    else:
        _LOGGER.info("Reusing precomputed mask artifacts for tile %s", tile_name)
        probability_volume = mask_artifacts.probability_volume
        mask = mask_artifacts.mask_low_res

    if mask.shape == full_res.shape:
        _LOGGER.info("Mask already at full resolution, skipping upscaling.")
        mask_upscaled = mask
    else:
        _LOGGER.info(f"Upscaling mask to full resolution: {full_res.shape}")
        mask_upscaled = upscale_mask_nearest(
            mask,
            full_res.shape,
            chunks=(128, 256, 256),
        )

    mask_path = out_mask_path.rstrip("/") + f"/{tile_name}"
    store_ome_zarr(
        mask_upscaled.astype(np.uint8),
        mask_path,
        n_levels,
        coordinate_transformations["scale"][-3:],
        coordinate_transformations["translation"],
        overwrite=overwrite,
        write_empty_chunks=False,
    )

    med_factor = (
        config.get("med_factor_binned")
        if is_binned_channel
        else config.get("med_factor_unbinned")
    )

    _LOGGER.info(f"Clipping low_res with median factor: {med_factor}")
    # Dask implementation can only compute nanmedian along subsets of axes at a time,
    # so we compute the nanmedian in two steps.
    nan_med = da.nanmedian(da.nanmedian(da.where(mask, low_res, np.nan), axis=(0,1)), axis=0).compute()
    # This fails with an OOM
    # nan_med = da.nanmedian(da.where(mask, low_res, np.nan), axis=tuple(d for d in low_res.ndim)).compute()
    _LOGGER.info(f"Computed median of tile foreground: {nan_med}")

    # Clamp the intensity values to reduce the impact of very bright neurites on the profile fit
    low_res_clipped = da.clip(low_res * mask, 0, nan_med * med_factor).astype(np.uint16).compute()

    del low_res, nan_med

    profile_sigma = config.get("profile_sigma", config.get("gaussian_sigma"))
    profile_percentile = config.get("profile_percentile", 75)
    profile_min_voxels = config.get("profile_min_voxels", 0)

    _LOGGER.info(
        "Computing 3D masked profiles (sigma=%s, percentile=%s, min_voxels=%s)",
        profile_sigma,
        profile_percentile,
        profile_min_voxels,
    )
    norm_x, median_xy = masked_axis_profile(
        low_res_clipped,
        mask,
        axis=2,
        smooth_sigma=profile_sigma,
        percentile=profile_percentile,
        min_voxels=profile_min_voxels,
    )
    _LOGGER.info("Global median from 3D masked profile (axis=X): %s", median_xy)

    fit_x = rescale_spline(
        np.arange(norm_x.size, dtype=np.float32),
        norm_x,
        full_res.shape[2],
        smoothing=config.get("spline_smoothing"),
    )
    limits_xy = config.get("limits_xy")
    if limits_xy is not None:
        fit_x = np.clip(fit_x, limits_xy[0], limits_xy[1])

    corrected = full_res
    fit_z = None

    if median_xy != 0:
        correction_x = fit_x.reshape(1, 1, -1)
        corrected = da.where(mask_upscaled, full_res / correction_x, full_res)

        norm_z, median_yz = masked_axis_profile(
            low_res_clipped,
            mask,
            axis=0,
            smooth_sigma=profile_sigma,
            percentile=profile_percentile,
            min_voxels=profile_min_voxels,
        )
        _LOGGER.info("Global median from 3D masked profile (axis=Z): %s", median_yz)

        fit_z = rescale_spline(
            np.arange(norm_z.size, dtype=np.float32),
            norm_z,
            full_res.shape[0],
            smoothing=config.get("spline_smoothing"),
        )
        limits_z = config.get("limits_z")
        if limits_z is not None:
            fit_z = np.clip(fit_z, limits_z[0], limits_z[1])

        correction_z = fit_z.reshape(-1, 1, 1)
        corrected = da.where(
            mask_upscaled, corrected / correction_z, corrected
        )

        # Use defaults consistent with get_fitting_config()
        global_factor = (
            config.get("global_factor_binned")
            if is_binned_channel
            else config.get("global_factor_unbinned")
        )
        ratio = global_factor / median_xy
        _LOGGER.info(
            "Doing global correction with factor: %s and median_xy: %s, ratio = %s",
            global_factor,
            median_xy,
            ratio,
        )
        corrected = da.where(
            mask_upscaled, corrected * ratio, corrected
        )
        corrected = da.clip(corrected, 0, 2**16 - 1)

    # Return corrected image and QC/debug artifacts for saving in main
    return (
        corrected,
        fit_x,
        fit_z,
        mask_artifacts,
    )


def save_metadata(
    data_process, out_path: str, tile_name: str, tile_path: str, results_dir: str
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
        Path(results_dir)
        / f"process_{Path(out_path).parent.name}_{tile_name}.json"
    )
    with open(process_json_path, "w") as f:
        f.write(process_json)

    input_metadata_path = get_parent_s3_path(get_parent_s3_path(tile_path))
    output_metadata_path = get_parent_s3_path(out_path)
    metadata_json_path = str(
        Path(results_dir)
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
        "med_factor_unbinned": 2,
        "percentile": 99,
        "gaussian_sigma": 2,
        "spline_smoothing": 0,
        "limits_xy": (0.25, 1.2),
        "limits_z": (0.25, 1.2),
        "global_factor_binned": 3200,
        "global_factor_unbinned": 100,
        "mask_probability_threshold": 0.9,
        "mask_probability_min_size": 10000,
        "gmm_n_components": 3,
        "gmm_max_samples": 2_000_000,
        "gmm_batch_size": 200_000,
        "gmm_random_state": 0,
    }


def save_method_outputs(
    method: str,
    tile_name: str,
    results_dir: str,
    save_outputs: bool,
    bkg: np.ndarray | None = None,
    bkg_slices: np.ndarray | None = None,
    artifacts: dict | None = None,
) -> None:
    """
    Save method-specific QC/debug outputs.

    Parameters
    ----------
    method : str
        The correction method used (e.g., 'fitting', 'basicpy', 'reference').
    tile_name : str
        Current tile name for filenames.
    results_dir : str
        Directory to write outputs.
    save_outputs : bool
        Gate to enable/disable saving.
    bkg : np.ndarray or None
        Background image (if computed).
    bkg_slices : np.ndarray or None
        Stack of background-dominated slices used for GMM fitting.
    artifacts : dict or None
        Method-specific artifacts; for 'fitting' expects keys:
          probability_volume, xy_proj, yz_proj,
          fit_x, fit_z.
    """
    if not save_outputs or results_dir is None:
        return

    # Always try to save background-derived outputs if provided
    try:
        if bkg_slices is not None:
            tifffile.imwrite(
                os.path.join(results_dir, f"{tile_name}_bkg_slices.tif"),
                np.asarray(bkg_slices, dtype=np.float32),
                imagej=True,
            )
        elif bkg is not None:
            tifffile.imwrite(
                os.path.join(results_dir, f"{tile_name}_bkg.tif"),
                np.asarray(bkg, dtype=np.float32),
                imagej=True,
            )
    except Exception:
        _LOGGER.exception("Failed saving background TIFF")

    if method == "fitting" and artifacts:
        try:
            fit_x = artifacts.get("fit_x")
            fit_z = artifacts.get("fit_z")
            if fit_x is not None:
                save_correction_curve_plot(
                    fit_x,
                    title=f"XY correction curve: {tile_name}",
                    xlabel="X (pixels)",
                    ylabel="Correction factor",
                    out_png=os.path.join(results_dir, f"{tile_name}_corr_xy.png"),
                )
            if fit_z is not None:
                save_correction_curve_plot(
                    fit_z,
                    title=f"YZ correction curve: {tile_name}",
                    xlabel="Z (slices)",
                    ylabel="Correction factor",
                    out_png=os.path.join(results_dir, f"{tile_name}_corr_yz.png"),
                )
        except Exception:
            _LOGGER.exception("Failed saving correction curve plots for fitting")

    # Placeholders for other methods; can be extended later
    elif method in ("basicpy", "reference"):
        _LOGGER.debug(f"No additional QC artifacts to save for method: {method}")


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


def set_dask_config(results_dir: str):
    """
    Set Dask configuration for memory management and temporary directory.
    """
    dask_tmp_dir = os.path.join(results_dir, "dask-tmp")
    dask.config.set(
        {
            "temporary-directory": dask_tmp_dir,
            "distributed.worker.memory.target": 0.7,
            "distributed.worker.memory.spill": 0.8,
            "distributed.worker.memory.pause": 0.9,
            "distributed.worker.memory.terminate": 0.95,
            "distributed.scheduler.allowed-failures": 10,
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
    parser.add_argument("--save-outputs", default=False, action="store_true", help="Save intermediate output files.")
    parser.add_argument("--results-dir", type=str, default=os.path.join(os.getcwd(), "results"), help="Directory to save results and metadata.")
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
    parser.add_argument("--use-reference-bkg", action="store_true", default=False, help="Use reference background image from S3 instead of estimating background.")
    parser.add_argument("--is-binned", action="store_true", default=False)
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


def create_mask_path(out_zarr_path: str) -> str:
    """
    Create the corresponding output mask path for 
    the input zarr image.

    Parameters
    -------
    out_zarr_path: str
        The output path for the corrected zarr dataset

    Returns
    -------
    str
        The mask path for the corrected zarr dataset
    """
    out_zarr_folder = Path(out_zarr_path).name
    out_mask_path = out_zarr_path.replace(out_zarr_folder, f"mask/{out_zarr_folder}")
    return out_mask_path


def create_probability_path(out_zarr_path: str) -> str:
    """Derive the output path for storing probability volumes as OME-Zarr."""

    out_zarr_folder = Path(out_zarr_path).name
    return out_zarr_path.replace(out_zarr_folder, f"probability/{out_zarr_folder}")


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

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # TODO: make a parameter
    binned_res = "0"

    set_dask_config(results_dir)
    co_memory = get_mem_limit()
    _LOGGER.info(f"CO_MEMORY: {co_memory}")

    # Support CO_MEMORY='auto' or explicit integer bytes
    if isinstance(co_memory, int):
        if co_memory <= 0:
            raise ValueError(
                "CO_MEMORY must be set to a positive integer value "
                "to allocate memory for Dask workers."
            )
        memory_limit = int(co_memory / max(1, args.num_workers))
    else:
        memory_limit = "auto"

    client = Client(
        LocalCluster(
            processes=True,
            n_workers=args.num_workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
        )
    )
    _LOGGER.info(f"Dask client: {client}")

    out_mask_path = create_mask_path(out_path)
    out_probability_path = create_probability_path(out_path)

    artifacts_destination = None
    tile_name_for_artifacts = Path(tile_paths[0]).name if tile_paths else None
    if tile_name_for_artifacts and out_path.startswith("s3://"):
        try:
            parent_output = get_parent_s3_path(out_path)
            artifacts_destination = f"{parent_output}/artifacts/{tile_name_for_artifacts}"
        except ValueError:
            _LOGGER.warning(
                "Unable to determine artifacts destination from output path %s",
                out_path,
            )

    start_date_time = datetime.now()
    data_process = create_processing_metadata(
        args, tile_paths[0], out_path, start_date_time, res
    )

    mask_artifacts: MaskArtifacts | None = None

    for tile_path in tile_paths:
        tile_name = Path(tile_path).name
        _LOGGER.info(f"Processing tile: {tile_name}")
        with performance_report(
            filename=os.path.join(results_dir, f"dask-report_{tile_name}.html")
        ):
            try:
                if args.is_binned:
                    is_binned_channel = True
                    resolution = "0"
                else:
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
                bkg_slices = None
                if not args.skip_bkg_sub:
                    _LOGGER.info("Performing background subtraction")
                    full_res, bkg, bkg_slices = background_subtraction(
                        tile_path, full_res, z, is_binned_channel, args.use_reference_bkg
                    )
                    # Background QC saving is centralized at the end per method

                if not args.skip_flat_field:
                    if method == "reference":
                        corrected = flatfield_reference(
                            full_res, 
                            args.flatfield_path
                        )
                    elif method == "basicpy":
                        corrected = flatfield_basicpy(
                            full_res, 
                            z, 
                            is_binned_channel, 
                            bkg, 
                            args.mask_dir, 
                            tile_name,
                            results_dir=results_dir
                        )
                    elif method == "fitting":
                        fitting_config = get_fitting_config()
                        corrected, fit_x, fit_z, mask_artifacts = flatfield_fitting(
                            full_res,
                            z,
                            is_binned_channel,
                            args.mask_dir,
                            tile_name,
                            out_mask_path,
                            out_probability_path,
                            coordinate_transformations,
                            args.overwrite,
                            args.n_levels,
                            fitting_config,
                            results_dir=results_dir,
                            bkg_slices=bkg_slices,
                            mask_artifacts=mask_artifacts,
                        )
                    else:
                        _LOGGER.error(f"Invalid method: {method}")
                        raise ValueError(f"Invalid method: {method}")
                else:
                    corrected = full_res

                corrected = corrected.astype(np.uint16)
                _LOGGER.info(f"Corrected array dtype: {corrected.dtype}")

                t0 = time.time()
                _LOGGER.info(f"Storing OME-Zarr for tile {tile_name}")
                store_ome_zarr(
                    corrected,
                    out_path.rstrip("/") + f"/{tile_name}",
                    args.n_levels,
                    coordinate_transformations["scale"][-3:],
                    coordinate_transformations["translation"],
                    overwrite=args.overwrite,
                    write_empty_chunks=True
                )
                _LOGGER.info(f"Storing OME-Zarr took {time.time() - t0:.2f}s")

                # Centralized, method-aware saving of QC outputs
                artifacts = None
                if method == "fitting":
                    artifacts = {
                        "fit_x": fit_x,
                        "fit_z": fit_z,
                    }
                save_method_outputs(
                    method,
                    tile_name,
                    results_dir,
                    args.save_outputs,
                    bkg=bkg,
                    bkg_slices=bkg_slices,
                    artifacts=artifacts,
                )

                data_process.end_date_time = datetime.now()
                save_metadata(data_process, out_path, tile_name, tile_path, results_dir)
            except Exception as e:
                _LOGGER.error(
                    f"Error processing tile {tile_name}: {e}", exc_info=True
                )
                raise

    if artifacts_destination:
        try:
            _LOGGER.info(
                "Uploading artifacts from %s to %s",
                results_dir,
                artifacts_destination,
            )
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--recursive",
                    results_dir,
                    artifacts_destination,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            _LOGGER.error(
                "Failed to upload artifacts to %s",
                artifacts_destination,
                exc_info=True,
            )
            raise


if __name__ == "__main__":
    main()
