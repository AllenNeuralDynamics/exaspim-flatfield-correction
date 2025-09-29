import os
import re
import glob
import time
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from numcodecs import blosc

blosc.use_threads = False
import tifffile
import dask
import dask.array as da
from dask.distributed import performance_report
from distributed import Client, LocalCluster
from dask_image.ndfilters import gaussian_filter as gaussian_filter_dask

from exaspim_flatfield_correction.basic import (
    fit_basic,
    transform_basic,
)
from exaspim_flatfield_correction.fitting import (
    apply_axis_corrections,
    compute_axis_fits,
)
from exaspim_flatfield_correction.background import (
    estimate_bkg, 
    subtract_bkg,
)
from exaspim_flatfield_correction.utils.mask_utils import (
    calc_gmm_prob,
    size_filter,
    upscale_mask_nearest,
    get_mask,
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
from exaspim_flatfield_correction.config import (
    FittingConfig,
    load_fitting_config,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)

DEFAULT_MEDIAN_SUMMARY_PATH = Path("/data/median_intensity_summary.json")


@dataclass
class MaskArtifacts:
    mask_low_res: da.Array


def read_median_intensity_summary(path: Path) -> dict[str, float]:
    """Load per-channel normalization targets from disk.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path pointing to ``median_intensity_summary.json``.

    Returns
    -------
    dict of str to float
        Mapping from channel identifiers to their ``mean_of_medians`` value.

    Notes
    -----
    The function logs and skips channels that are missing or have
    non-numeric ``mean_of_medians`` entries.
    """
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except Exception:
        _LOGGER.exception("Failed to read median intensity summary at %s", path)
        return {}

    overrides: dict[str, float] = {}
    for channel, payload in data.items():
        mean_value = None
        if isinstance(payload, dict):
            mean_value = payload.get("mean_of_medians")
        if mean_value is None:
            _LOGGER.warning(
                "Skipping channel %s in %s: missing mean_of_medians",
                channel,
                path,
            )
            continue
        try:
            overrides[str(channel)] = float(mean_value)
        except (TypeError, ValueError):
            _LOGGER.warning(
                "Skipping channel %s in %s: mean_of_medians=%r is not numeric",
                channel,
                path,
                mean_value,
            )
    return overrides


def apply_median_summary_override(
    fitting_config: FittingConfig,
    median_summary: dict[str, float],
    tile_name: str,
    *,
    is_binned_channel: bool,
    binned_channel: str | None,
) -> None:
    """Update global correction factors using the summary file when possible.

    Parameters
    ----------
    fitting_config : FittingConfig
        Mutable fitting configuration to update in-place.
    median_summary : dict of str to float
        Channel-to-``mean_of_medians`` mapping loaded from disk.
    tile_name : str
        Name of the tile currently being processed.
    is_binned_channel : bool
        Whether the tile corresponds to the binned channel.
    binned_channel : str or None
        Identifier of the binned channel, when known.
    """
    if not median_summary:
        return

    override_value: float | None = None
    binned_key = str(binned_channel) if binned_channel else None

    if is_binned_channel and binned_key:
        override_value = median_summary.get(binned_key)
    elif not is_binned_channel:
        tile_lower = tile_name.lower()
        for channel, value in median_summary.items():
            if channel == binned_key:
                continue
            if channel.lower() in tile_lower:
                override_value = value
                break
        if override_value is None:
            for channel, value in median_summary.items():
                if channel != binned_key:
                    override_value = value
                    break

    if override_value is None:
        return

    if is_binned_channel:
        if fitting_config.global_factor_binned != override_value:
            _LOGGER.info(
                "Overriding binned global factor with %.4f", override_value
            )
        fitting_config.global_factor_binned = override_value
    else:
        if fitting_config.global_factor_unbinned != override_value:
            _LOGGER.info(
                "Overriding unbinned global factor with %.4f", override_value
            )
        fitting_config.global_factor_unbinned = override_value


def get_mem_limit() -> int | str:
    """Return the memory limit to pass into ``LocalCluster``.

    Returns
    -------
    int or str
        Explicit byte count if ``CO_MEMORY`` is set, otherwise ``"auto"``.

    Raises
    ------
    ValueError
        If ``CO_MEMORY`` is defined but cannot be parsed as an integer.
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


def parse_inputs(args: argparse.Namespace) -> dict[str, str | list[str] | None]:
    """Collect pipeline inputs from CLI arguments and optional metadata.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict of str to (str, list[str], or None)
        Mapping that includes the tile paths, output destination, method,
        resolution, and binned-channel identifier (when present).

    Raises
    ------
    ValueError
        If no metadata file can be located when ``--zarr`` is omitted.
    """
    if args.zarr != "":
        tile_paths = [args.zarr]
        out_path = args.output
        method = args.method
        res = str(args.res)
        binned_channel = None
        if args.is_binned:
            tile_name = Path(args.zarr).name
            inferred_channel = extract_channel_from_tile_name(tile_name)
            if inferred_channel is not None:
                binned_channel = inferred_channel
            else:
                _LOGGER.warning(
                    "Unable to infer channel number from tile %s despite --is-binned",
                    tile_name,
                )
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
    params: dict[str, str | list[str] | None] = {
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
    """Remove background estimates from the full-resolution volume.

    Parameters
    ----------
    tile_path : str
        Filesystem or S3 path to the tile Zarr group.
    full_res : dask.array.Array
        Full-resolution image volume to correct (modified lazily).
    z : zarr.hierarchy.Group
        Open Zarr hierarchy for the tile.
    is_binned_channel : bool, default=False
        Flag indicating whether the tile represents the binned channel.
    use_reference_bkg : bool, default=False
        When ``True`` load a static reference background, otherwise estimate
        it from the data.

    Returns
    -------
    tuple
        ``(full_res_corrected, background_image, background_slices)`` where
        ``background_slices`` may be ``None`` if no probabilistic model is
        required.
    """
    if use_reference_bkg:
        bkg_path = get_bkg_path(tile_path)
        bkg = read_bkg_image(bkg_path).astype(np.float32)
        bkg_slices = None
    else:
        bkg_res = "0" if is_binned_channel else "3"
        _LOGGER.info(f"Using resolution {bkg_res} for background estimation")
        low_res = da.from_zarr(z[bkg_res]).squeeze().astype(np.float32)
        bkg, bkg_slices = estimate_bkg(
            gaussian_filter_dask(low_res, sigma=1).compute()
        )

    full_res = subtract_bkg(
        full_res,
        da.from_array(
            resize(bkg, full_res.shape[1:]), chunks=full_res.chunksize[1:]
        ),
    )
    return full_res, bkg, bkg_slices


def flatfield_reference(full_res: da.Array, flatfield_path: str) -> da.Array:
    """Apply reference flatfield correction using a precomputed image.

    Parameters
    ----------
    full_res : dask.array.Array
        Full-resolution volume to be corrected.
    flatfield_path : str
        Location of the reference flatfield image (local path or S3 URL).

    Returns
    -------
    dask.array.Array
        Corrected image volume with the flatfield applied.

    Raises
    ------
    ValueError
        If ``flatfield_path`` is not provided.
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
    bkg: np.ndarray | None = None,
    mask_dir: str | None = None,
    tile_name: str | None = None,
    max_slices: int = 100,
    working_size: int = 512,
    sort_intensity: bool = True,
    shuffle_frames: bool = False,
    autotune: bool = False,
    results_dir: str | None = None,
) -> da.Array:
    """Apply BasicPy flatfield correction to the supplied volume.

    Parameters
    ----------
    full_res : dask.array.Array
        Full-resolution image volume to correct.
    z : zarr.hierarchy.Group
        Zarr hierarchy for the tile.
    is_binned_channel : bool
        ``True`` when processing the binned channel.
    bkg : numpy.ndarray or None, default=None
        Optional background image to subtract prior to fitting.
    mask_dir : str or None, default=None
        Directory containing an externally supplied mask for the tile.
    tile_name : str or None, default=None
        Tile identifier used when saving intermediate artifacts.
    max_slices : int, default=100
        Maximum number of slices to pass into BasicPy.
    working_size : int, default=512
        Working size for BasicPy processing.
    sort_intensity : bool, default=True
        Whether to sort frames by intensity before fitting.
    shuffle_frames : bool, default=False
        Whether to shuffle the frames randomly prior to fitting.
    autotune : bool, default=False
        Enable BasicPy autotuning to determine optimal parameters.
    results_dir : str or None, default=None
        Destination directory for any intermediate mask artifacts.

    Returns
    -------
    dask.array.Array
        Lazily evaluated corrected volume.
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
            tile_name,
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


def _preprocess_mask(
    mask: np.ndarray,
    low_res_shape: tuple[int, int, int],
    results_dir: str,
    tile_name: str,
) -> da.Array:
    """Normalize mask shape and persist it as a reusable Zarr array.

    Parameters
    ----------
    mask : numpy.ndarray
        Input mask to be upscaled and stored.
    low_res_shape : tuple of int
        Expected shape that matches the low-resolution volume.
    results_dir : str
        Directory where the temporary Zarr mask should be written.
    tile_name : str
        Identifier used when naming the persisted mask.

    Returns
    -------
    dask.array.Array
        Dask array backed by the stored mask Zarr.
    """
    mask_name = str(Path(results_dir) / f"{tile_name}_mask_low_res.zarr")
    if mask.shape != low_res_shape:
        mask = upscale_mask_nearest(
            da.from_array(mask, chunks=(128, 256, 256)),
            low_res_shape,
            chunks=(128, 256, 256),
        ).compute()
    mask = mask.astype(np.uint8)
    zarr.save_array(
        str(mask_name),
        mask,
        chunks=(128, 256, 256),
        compressor=blosc.Blosc(cname="zstd", clevel=1),
    )
    return da.from_zarr(mask_name, chunks=(128, 256, 256))


def _create_mask_artifacts(
    low_res: da.Array,
    z: zarr.hierarchy.Group,
    fitting_res: str,
    mask_dir: str,
    tile_name: str,
    results_dir: str,
    config: FittingConfig,
    bkg_slices: np.ndarray | None,
    out_probability_path: str,
    overwrite: bool,
) -> MaskArtifacts:
    """Generate mask artifacts and optional probability volumes for a tile.

    Parameters
    ----------
    low_res : dask.array.Array
        Low-resolution stack used for mask inference.
    z : zarr.hierarchy.Group
        Zarr hierarchy that contains the tile data.
    fitting_res : str
        Resolution key within ``z`` from which the mask is derived.
    mask_dir : str
        Directory containing raw mask files.
    tile_name : str
        Identifier of the tile driving mask generation.
    results_dir : str
        Destination directory for persisted mask artifacts.
    config : FittingConfig
        Configuration controlling mask refinement and thresholds.
    bkg_slices : numpy.ndarray or None
        Background-dominant slices used for GMM refinement when available.
    out_probability_path : str
        Root path where probability volumes are stored.
    overwrite : bool
        Whether existing outputs may be replaced.

    Returns
    -------
    MaskArtifacts
        Container with the low-resolution mask suitable for reuse.

    Raises
    ------
    ValueError
        If probability refinement is requested without ``bkg_slices``.
    """
    _LOGGER.info("Creating mask artifacts using tile %s", tile_name)
    initial_mask = _preprocess_mask(
        size_filter(load_mask_from_dir(mask_dir, tile_name), k_largest=1, min_size=None),
        low_res.shape,
        results_dir,
        tile_name,
    ).astype(bool)

    mask_low_res = initial_mask

    if config.enable_gmm_refinement:
        if bkg_slices is None:
            raise ValueError(
                "Background slices are required to fit the mask GMM on the reference tile"
            )

        gmm_n_components = config.gmm_n_components
        gmm_max_samples = config.gmm_max_samples
        gmm_batch_size = config.gmm_batch_size
        gmm_random_state = config.gmm_random_state

        _LOGGER.info(
            "Fitting GMM probabilities with n_components=%d, max_samples=%d, batch_size=%d",
            gmm_n_components,
            gmm_max_samples,
            gmm_batch_size,
        )

        probability_volume = (
            calc_gmm_prob(
                gaussian_filter_dask(low_res.astype(np.float32), sigma=1),
                initial_mask,
                da.from_array(bkg_slices, chunks=(64, 64, 64)),
                n_components_fg=gmm_n_components,
                n_components_bg=gmm_n_components,
                max_samples_fg=gmm_max_samples,
                max_samples_bg=gmm_max_samples,
                random_state=gmm_random_state,
                erosion_radius=config.erosion_radius,
            )
            .astype(np.float32)
        )

        probability_path = out_probability_path.rstrip("/") + f"/{tile_name}"
        probability_transformations = parse_ome_zarr_transformations(
            z, fitting_res
        )
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
        probability_volume = da.from_zarr(
            probability_path, component="0"
        ).squeeze().compute()

        prob_threshold = config.mask_probability_threshold
        prob_min_size = config.mask_probability_min_size
        _LOGGER.info(
            "Thresholding probability volume with threshold %.3f and min_size %s",
            prob_threshold,
            prob_min_size,
        )
        mask_low_res = get_mask(
            probability_volume,
            threshold=prob_threshold,
            min_size=prob_min_size,
            k_largest=1
        )
        mask_low_res = (
            _preprocess_mask(mask_low_res, low_res.shape, results_dir, tile_name)
            .astype(bool)
        )

    return MaskArtifacts(
        mask_low_res=mask_low_res,
    )


def flatfield_fitting(
    full_res: da.Array,
    z: zarr.hierarchy.Group,
    is_binned_channel: bool,
    mask_dir: str,
    tile_name: str,
    out_mask_path: str,
    out_probability_path: str,
    coordinate_transformations: dict[str, Any],
    overwrite: bool,
    n_levels: int,
    config: FittingConfig,
    results_dir: str | None = None,
    bkg_slices: np.ndarray | None = None,
    mask_artifacts: MaskArtifacts | None = None,
) -> tuple[da.Array, dict[str, np.ndarray], MaskArtifacts | None]:
    """Run the fitting-based flatfield workflow for a single tile.

    Parameters
    ----------
    full_res : dask.array.Array
        Full-resolution volume to correct.
    z : zarr.hierarchy.Group
        Zarr hierarchy that stores the tile data.
    is_binned_channel : bool
        Indicates whether the tile corresponds to the binned channel.
    mask_dir : str
        Directory containing the initial foreground mask(s).
    tile_name : str
        Identifier of the tile being processed.
    out_mask_path : str
        Base path where upscaled masks should be written.
    out_probability_path : str
        Base path where probability volumes are persisted.
    coordinate_transformations : dict of str to Any
        Transformation metadata copied into the OME-NGFF output.
    overwrite : bool
        Whether existing OME-Zarr outputs may be replaced.
    n_levels : int
        Number of pyramid levels to generate for outputs.
    config : FittingConfig
        Configuration controlling fitting behaviour and thresholds.
    results_dir : str or None, default=None
        Directory used for storing intermediate artifacts.
    bkg_slices : numpy.ndarray or None, default=None
        Background-dominant slices leveraged by the optional GMM refinement.
    mask_artifacts : MaskArtifacts or None, default=None
        Previously computed mask artifacts to reuse across tiles.

    Returns
    -------
    tuple
        ``(corrected_volume, axis_fits, mask_artifacts)`` where
        ``axis_fits`` maps ``{"x", "y", "z"}`` to their correction curves.
    """
    fitting_res = "0" if is_binned_channel else "3"
    low_res = da.from_zarr(z[fitting_res]).squeeze().astype(np.float32)

    if mask_artifacts is None:
        mask_artifacts = _create_mask_artifacts(
            low_res=low_res,
            z=z,
            fitting_res=fitting_res,
            mask_dir=mask_dir,
            tile_name=tile_name,
            results_dir=results_dir,
            config=config,
            bkg_slices=bkg_slices,
            out_probability_path=out_probability_path,
            overwrite=overwrite,
        )
    else:
        _LOGGER.info(
            "Reusing precomputed mask artifacts for tile %s", tile_name
        )

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
    # Re-read the computed mask back from S3 for performance
    mask_upscaled = da.from_zarr(mask_path, component="0").squeeze()

    low_res = subtract_bkg(
        low_res,
        da.from_array(
            resize(np.median(bkg_slices, axis=0).astype(np.float32), low_res.shape[1:]), 
            chunks=low_res.chunksize[1:]
        ),
    )

    med_factor = (
        config.med_factor_binned
        if is_binned_channel
        else config.med_factor_unbinned
    )

    _LOGGER.info(f"Clipping low_res with median factor: {med_factor}")
    # Dask implementation can only compute nanmedian along subsets of axes at a time,
    # so we compute the nanmedian in two steps.
    nan_med = da.nanmedian(
        da.nanmedian(da.where(mask, low_res, np.nan), axis=(0, 1)), axis=0
    ).compute()
    # This fails with an OOM
    # nan_med = da.nanmedian(da.where(mask, low_res, np.nan), axis=tuple(d for d in low_res.ndim)).compute()
    _LOGGER.info(f"Computed median of tile foreground: {nan_med}")

    # Clamp the intensity values to reduce the impact of very bright neurites on the profile fit
    low_res_clipped = (
        da.clip(low_res * mask, 0, nan_med * med_factor)
        .astype(np.uint16)
    )

    del low_res

    profile_sigma = config.gaussian_sigma
    profile_percentile = config.profile_percentile
    profile_min_voxels = config.profile_min_voxels
    spline_smoothing = config.spline_smoothing

    _LOGGER.info(
        "Computing masked profiles (sigma=%s, percentile=%s, min_voxels=%s)",
        profile_sigma,
        profile_percentile,
        profile_min_voxels,
    )
    axis_fits, axis_medians = compute_axis_fits(
        low_res_clipped,
        mask,
        full_res.shape,
        smooth_sigma=profile_sigma,
        percentile=profile_percentile,
        min_voxels=profile_min_voxels,
        spline_smoothing=spline_smoothing,
        limits_x=config.limits_x,
        limits_y=config.limits_y,
        limits_z=config.limits_z,
    )

    median_xy = axis_medians["x"]
    median_xz = axis_medians["y"]
    median_yz = axis_medians["z"]
    _LOGGER.info(
        "Global median from 3D masked profile (axis=X): %s", median_xy
    )
    _LOGGER.info(
        "Global median from 3D masked profile (axis=Y): %s", median_xz
    )
    _LOGGER.info(
        "Global median from 3D masked profile (axis=Z): %s", median_yz
    )

    global_factor = (
        config.global_factor_binned
        if is_binned_channel
        else config.global_factor_unbinned
    )

    corrected = apply_axis_corrections(
        full_res,
        mask_upscaled,
        axis_fits,
        axis_medians,
        global_factor=global_factor,
    )

    # Return corrected image and QC/debug artifacts for saving in main
    return (
        corrected,
        axis_fits,
        mask_artifacts,
    )


def save_metadata(
    data_process: Any,
    out_path: str,
    tile_name: str,
    tile_path: str,
    results_dir: str,
) -> None:
    """Persist processing metadata alongside the corrected tile outputs.

    Parameters
    ----------
    data_process : Any
        Structured metadata object exposing ``model_dump_json``.
    out_path : str
        Target path of the corrected tile Zarr store.
    tile_name : str
        Identifier of the tile being processed.
    tile_path : str
        Source Zarr path for the tile.
    results_dir : str
        Directory in which metadata JSON artifacts are saved.
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


def save_method_outputs(
    method: str,
    tile_name: str,
    results_dir: str,
    save_outputs: bool,
    bkg: np.ndarray | None = None,
    bkg_slices: np.ndarray | None = None,
    artifacts: dict[str, Any] | None = None,
) -> None:
    """Persist optional QC artifacts for the specified correction method.

    Parameters
    ----------
    method : str
        Correction method identifier (e.g., ``"fitting"``).
    tile_name : str
        Name of the tile being processed.
    results_dir : str
        Directory where QC artifacts should be written.
    save_outputs : bool
        Flag that enables or skips artifact persistence.
    bkg : numpy.ndarray or None, default=None
        Estimated background image for the tile.
    bkg_slices : numpy.ndarray or None, default=None
        Background-dominated slices used to fit probabilistic models.
    artifacts : dict of str to Any or None, default=None
        Additional method-specific artifacts to serialize.
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
            axis_fits = artifacts.get("axis_fits")
            if axis_fits and "x" in axis_fits:
                save_correction_curve_plot(
                    axis_fits["x"],
                    title=f"XY correction curve: {tile_name}",
                    xlabel="X (pixels)",
                    ylabel="Correction factor",
                    out_png=os.path.join(
                        results_dir, f"{tile_name}_corr_xy.png"
                    ),
                )
            if axis_fits and "y" in axis_fits:
                save_correction_curve_plot(
                    axis_fits["y"],
                    title=f"XZ correction curve: {tile_name}",
                    xlabel="Y (pixels)",
                    ylabel="Correction factor",
                    out_png=os.path.join(
                        results_dir, f"{tile_name}_corr_xz.png"
                    ),
                )
            if axis_fits and "z" in axis_fits:
                save_correction_curve_plot(
                    axis_fits["z"],
                    title=f"YZ correction curve: {tile_name}",
                    xlabel="Z (slices)",
                    ylabel="Correction factor",
                    out_png=os.path.join(
                        results_dir, f"{tile_name}_corr_yz.png"
                    ),
                )
        except Exception:
            _LOGGER.exception(
                "Failed saving correction curve plots for fitting"
            )

    # Placeholders for other methods; can be extended later
    elif method in ("basicpy", "reference"):
        _LOGGER.debug(
            f"No additional QC artifacts to save for method: {method}"
        )


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
    """Configure Dask memory thresholds and scratch directory.

    Parameters
    ----------
    results_dir : str
        Directory in which the Dask temporary folder will be created.
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
    parser.add_argument(
        "--save-outputs",
        default=False,
        action="store_true",
        help="Save intermediate output files.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(os.getcwd(), "results"),
        help="Directory to save results and metadata.",
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
    parser.add_argument(
        "--fitting-config",
        type=str,
        default=None,
        help="Path to a JSON file with fitting configuration overrides.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-level", type=str, default=logging.INFO)
    parser.add_argument(
        "--n-levels",
        type=int,
        default=1,
        help="Number of zarr pyramid levels (default: 1)",
    )
    parser.add_argument(
        "--use-reference-bkg",
        action="store_true",
        default=False,
        help="Use reference background image from S3 instead of estimating background.",
    )
    parser.add_argument("--is-binned", action="store_true", default=False)
    parser.add_argument(
        "--median-summary-path",
        type=str,
        default=str(DEFAULT_MEDIAN_SUMMARY_PATH),
        help=(
            "Path to JSON file containing per-channel mean_of_medians values "
            "for global normalization (default: /data/median_intensity_summary.json)."
        ),
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


def create_mask_path(out_zarr_path: str) -> str:
    """Derive the OME-Zarr mask output path from the corrected dataset path.

    Parameters
    ----------
    out_zarr_path : str
        Path to the corrected OME-Zarr dataset.

    Returns
    -------
    str
        Path pointing to the associated mask dataset.
    """
    out_zarr_folder = Path(out_zarr_path).name
    out_mask_path = out_zarr_path.replace(
        out_zarr_folder, f"mask/{out_zarr_folder}"
    )
    return out_mask_path


def create_probability_path(out_zarr_path: str) -> str:
    """Derive the probability volume output path from the corrected dataset.

    Parameters
    ----------
    out_zarr_path : str
        Path to the corrected OME-Zarr dataset.

    Returns
    -------
    str
        Path pointing to the probability-volume dataset.
    """

    out_zarr_folder = Path(out_zarr_path).name
    return out_zarr_path.replace(
        out_zarr_folder, f"probability/{out_zarr_folder}"
    )


def main() -> None:
    """Execute the flatfield correction pipeline for the requested tiles."""
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

    median_summary_path = Path(args.median_summary_path)

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
            artifacts_destination = (
                f"{parent_output}/artifacts/{tile_name_for_artifacts}"
            )
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
    fitting_config: FittingConfig | None = None
    median_summary: dict[str, float] = {}
    if method == "fitting":
        fitting_config = load_fitting_config(args.fitting_config)
        if args.fitting_config:
            _LOGGER.info("Loaded fitting config from %s", args.fitting_config)
        # dump to results folder
        fitting_config.to_file(
            os.path.join(results_dir, "fitting_config.json")
        )
        median_summary = read_median_intensity_summary(median_summary_path)
        if median_summary:
            _LOGGER.info(
                "Loaded median intensity overrides for channels: %s",
                ", ".join(sorted(median_summary)),
            )

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

                full_res = (
                    da.from_zarr(z[resolution]).squeeze().astype(np.float32)
                )
                _LOGGER.info(f"Full resolution array shape: {full_res.shape}")

                bkg = None
                bkg_slices = None
                if not args.skip_bkg_sub:
                    _LOGGER.info("Performing background subtraction")
                    full_res, bkg, bkg_slices = background_subtraction(
                        tile_path,
                        full_res,
                        z,
                        is_binned_channel,
                        args.use_reference_bkg,
                    )
                    # Background QC saving is centralized at the end per method

                axis_fits = None
                if not args.skip_flat_field:
                    if method == "reference":
                        corrected = flatfield_reference(
                            full_res, args.flatfield_path
                        )
                    elif method == "basicpy":
                        corrected = flatfield_basicpy(
                            full_res,
                            z,
                            is_binned_channel,
                            bkg,
                            args.mask_dir,
                            tile_name,
                            results_dir=results_dir,
                        )
                    elif method == "fitting":
                        if fitting_config is None:
                            raise ValueError(
                                "Fitting configuration failed to initialize"
                            )
                        apply_median_summary_override(
                            fitting_config,
                            median_summary,
                            tile_name,
                            is_binned_channel=is_binned_channel,
                            binned_channel=binned_channel,
                        )
                        corrected, axis_fits, mask_artifacts = (
                            flatfield_fitting(
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
                    write_empty_chunks=True,
                )
                _LOGGER.info(f"Storing OME-Zarr took {time.time() - t0:.2f}s")

                # Centralized, method-aware saving of QC outputs
                artifacts = None
                if method == "fitting":
                    artifacts = {
                        "axis_fits": axis_fits,
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
                save_metadata(
                    data_process, out_path, tile_name, tile_path, results_dir
                )
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
