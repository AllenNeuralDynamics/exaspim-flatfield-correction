import argparse
import glob
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import blosc

blosc.use_threads = False
import dask.array as da
import tifffile
from dask.distributed import performance_report
from dask_image.ndfilters import gaussian_filter as gaussian_filter_dask
from scipy.ndimage import binary_fill_holes

from exaspim_flatfield_correction.background import estimate_bkg, subtract_bkg
from exaspim_flatfield_correction.basic import fit_basic, transform_basic
from exaspim_flatfield_correction.config import (
    FittingConfig,
    apply_median_summary_override,
    load_fitting_config,
    read_median_intensity_summary,
)
from exaspim_flatfield_correction.fitting import (
    apply_axis_corrections,
    compute_axis_fits,
    calc_percentile_weight,
)
from exaspim_flatfield_correction.utils.mask_utils import (
    size_filter,
    upscale_mask_nearest,
)
from exaspim_flatfield_correction.utils.metadata_utils import (
    create_processing_metadata,
    save_metadata,
)
from exaspim_flatfield_correction.utils.utils import (
    array_chunks,
    chunks_2d,
    extract_channel_from_tile_name,
    get_bkg_path,
    get_parent_s3_path,
    load_mask_from_dir,
    read_bkg_image,
    resize,
    save_correction_curve_plot,
    upload_artifacts,
    weighted_percentile
)
from exaspim_flatfield_correction.utils.zarr_utils import (
    initialize_zarr_group,
    parse_ome_zarr_transformations,
    store_ome_zarr,
)
from exaspim_flatfield_correction.utils.dask_utils import create_dask_client


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


@dataclass
class MaskArtifacts:
    """Container for cached mask artifacts reused within the pipeline."""

    mask_low_res: da.Array
    probability_volume: da.Array | None = None


def resolve_args(
    args: argparse.Namespace,
) -> dict[str, str | list[str] | None]:
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
        ``(full_res_corrected, background_image, background_slice_indices)``
        where ``background_slice_indices`` may be ``None`` if no probabilistic
        model is required.
    """
    if use_reference_bkg:
        bkg_path = get_bkg_path(tile_path)
        bkg = read_bkg_image(bkg_path).astype(np.float32)
        bkg_slice_indices = None
    else:
        bkg_res = "0" if is_binned_channel else "3"
        _LOGGER.info(f"Using resolution {bkg_res} for background estimation")
        low_res = da.from_zarr(z[bkg_res]).squeeze().astype(np.float32)
        bkg, bkg_slice_indices = estimate_bkg(
            gaussian_filter_dask(low_res, sigma=1).compute()
        )

    full_res = subtract_bkg(
        full_res,
        da.from_array(
            resize(bkg, full_res.shape[1:]), chunks=full_res.chunksize[1:]
        ),
    )
    return full_res, bkg, bkg_slice_indices


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
    spatial_chunks = chunks_2d(full_res)
    flatfield = da.from_array(flatfield, chunks=spatial_chunks)
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
        mask_chunks = array_chunks(low_res)
        mask = _preprocess_mask(
            load_mask_from_dir(mask_dir, tile_name),
            low_res.shape,
            results_dir,
            tile_name,
            chunks=mask_chunks,
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
    *,
    chunks: tuple[int, ...],
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
    chunks : tuple of int
        Chunk specification matching the target low-resolution Dask array.

    Returns
    -------
    dask.array.Array
        Dask array backed by the stored mask Zarr.
    """
    mask_name = str(Path(results_dir) / f"{tile_name}_mask_low_res.zarr")
    if mask.shape != low_res_shape:
        mask = upscale_mask_nearest(
            da.from_array(mask, chunks=chunks),
            low_res_shape,
            chunks=chunks,
        ).compute()
    mask = mask.astype(np.uint8)
    zarr.save_array(
        str(mask_name),
        mask,
        chunks=chunks,
        compressor=blosc.Blosc(cname="zstd", clevel=1),
    )
    return da.from_zarr(mask_name)


def _create_mask_artifacts(
    low_res: da.Array,
    z: zarr.hierarchy.Group,
    fitting_res: str,
    mask_dir: str,
    tile_name: str,
    results_dir: str,
    config: FittingConfig,
    bkg_slice_indices: np.ndarray,
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
    bkg_slice_indices : numpy.ndarray
        Indices of background-dominant slices used when estimating background
        statistics for the probability volume.
    out_probability_path : str
        Root path where probability volumes are stored.
    overwrite : bool
        Whether existing outputs may be replaced.

    Returns
    -------
    MaskArtifacts
        Container with the low-resolution mask suitable for reuse.

    """
    _LOGGER.info("Creating mask artifacts using tile %s", tile_name)
    mask_chunks = array_chunks(low_res)
    try:
        initial_mask = load_mask_from_dir(mask_dir, tile_name)
    except FileNotFoundError as e:
        _LOGGER.warning(f"Mask does not exist for tile {tile_name}. Skipping correction.")
        return None
    initial_mask = _preprocess_mask(
        binary_fill_holes(
            size_filter(
                initial_mask, k_largest=2, min_size=None
        )),
        low_res.shape,
        results_dir,
        tile_name,
        chunks=mask_chunks,
    ).astype(bool)
    initial_mask = initial_mask & (low_res != 0)

    probability_volume: da.Array | None = None
    if config.enable_gmm_refinement:
        slice_indices = np.asarray(bkg_slice_indices, dtype=np.int64)
        bg_reference = da.take(low_res, slice_indices, axis=0).compute()

        _LOGGER.info(
            "Computing percentile-based probability weights (low=%.1f, high=%.1f, eps=%.3f)",
            config.probability_bg_low_percentile,
            config.probability_bg_high_percentile,
            config.probability_ramp_eps,
        )

        probability_volume = calc_percentile_weight(
            low_res,
            bg_reference,
            low_percentile=config.probability_bg_low_percentile,
            high_percentile=config.probability_bg_high_percentile,
            eps=config.probability_ramp_eps,
            smooth_sigma=config.probability_smooth_sigma,
            start_frac=config.probability_ramp_start_frac,
            nu=config.probability_ramp_nu,
        ).astype(np.float32)

        probability_path = out_probability_path.rstrip("/") + f"/{tile_name}"
        probability_transformations = parse_ome_zarr_transformations(
            z, fitting_res
        )
        probability_scale = probability_transformations["scale"]
        probability_translation = probability_transformations["translation"]

        _LOGGER.info("Storing probability volume at %s", probability_path)
        store_ome_zarr(
            probability_volume,
            probability_path,
            5,
            tuple(probability_scale[-3:]),
            tuple(probability_translation),
            overwrite=overwrite,
            write_empty_chunks=False,
        )
        # Do not materialize into memory until needed
        probability_volume = da.from_zarr(
            probability_path, component="0"
        ).squeeze()

    return MaskArtifacts(
        mask_low_res=initial_mask,
        probability_volume=probability_volume,
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
    bkg: np.ndarray,
    bkg_slice_indices: np.ndarray,
    results_dir: str | None = None,
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
    bkg : numpy.ndarray
        Estimated 2D background image used for subtraction of ``low_res``.
    bkg_slice_indices : numpy.ndarray
        Indices of background-dominant slices leveraged by the optional
        probability refinement.
    results_dir : str or None, default=None
        Directory used for storing intermediate artifacts.
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

    low_res = subtract_bkg(
        low_res,
        da.from_array(
            resize(bkg.astype(np.float32, copy=False), low_res.shape[1:]),
            chunks=low_res.chunksize[1:],
        ),
    )

    if mask_artifacts is None:
        mask_artifacts = _create_mask_artifacts(
            low_res=low_res,
            z=z,
            fitting_res=fitting_res,
            mask_dir=mask_dir,
            tile_name=tile_name,
            results_dir=results_dir,
            config=config,
            bkg_slice_indices=bkg_slice_indices,
            out_probability_path=out_probability_path,
            overwrite=overwrite,
        )
    else:
        _LOGGER.info(
            "Reusing precomputed mask artifacts for tile %s", tile_name
        )
    if mask_artifacts is None:
        _LOGGER.warning(f"mask_artifacts is None. Skipping correction for tile {tile_name}")
        return full_res, None, None

    mask = mask_artifacts.mask_low_res

    if mask.shape == full_res.shape:
        _LOGGER.info("Mask already at full resolution, skipping upscaling.")
        mask_upscaled = mask
    else:
        _LOGGER.info(f"Upscaling mask to full resolution: {full_res.shape}")
        mask_upscaled = upscale_mask_nearest(
            mask,
            full_res.shape,
            chunks=array_chunks(full_res),
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

    med_factor = (
        config.med_factor_binned
        if is_binned_channel
        else config.med_factor_unbinned
    )
    profile_sigma = config.gaussian_sigma
    profile_percentile = config.profile_percentile
    profile_min_voxels = config.profile_min_voxels
    spline_smoothing = config.spline_smoothing

    weights = mask_artifacts.probability_volume

    global_val = weighted_percentile(
        low_res.astype(np.uint16),
        mask,
        profile_percentile,
        weights=weights,
    )
    
    _LOGGER.info(
        f"Computed {profile_percentile} percentile of tile foreground: {global_val}"
    )

    if global_val == 0:
        _LOGGER.warning(
            f"Skipping flatfield correction for zero-mean tile {tile_name}"
        )   
        return full_res, None, mask_artifacts

    # Clamp the intensity values to reduce the impact of very bright neurites on the profile fit
    _LOGGER.info(f"Clipping low_res with median factor: {med_factor}")
    low_res = np.clip(low_res.compute(), 0, global_val * med_factor)

    _LOGGER.info(
        "Computing masked profiles (sigma=%s, percentile=%s, min_voxels=%s)",
        profile_sigma,
        profile_percentile,
        profile_min_voxels,
    )
    axis_fits = compute_axis_fits(
        gaussian_filter_dask(low_res, sigma=profile_sigma).compute(),
        mask,
        full_res.shape,
        percentile=profile_percentile,
        min_voxels=profile_min_voxels,
        spline_smoothing=spline_smoothing,
        limits_x=config.limits_x,
        limits_y=config.limits_y,
        limits_z=config.limits_z,
        weights=weights.compute() if weights is not None else None,
        global_med=global_val,
    )
    del low_res, mask, weights

    global_factor = (
        config.global_factor_binned
        if is_binned_channel
        else config.global_factor_unbinned
    )

    corrected = apply_axis_corrections(
        full_res,
        mask_upscaled,
        axis_fits,
        global_med=global_val,
        global_factor=global_factor,
        ratio_limits=config.global_ratio_limits,
    )

    # Return corrected image and QC/debug artifacts for saving in main
    return (
        corrected,
        axis_fits,
        mask_artifacts,
    )


def save_method_outputs(
    method: str,
    tile_name: str,
    results_dir: str,
    save_outputs: bool,
    bkg: np.ndarray | None = None,
    bkg_slice_indices: np.ndarray | None = None,
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
    bkg_slice_indices : numpy.ndarray or None, default=None
        Indices of background-dominated slices used to fit probabilistic
        models.
    artifacts : dict of str to Any or None, default=None
        Additional method-specific artifacts to serialize.
    """
    if not save_outputs or results_dir is None:
        return

    # Always try to save background-derived outputs if provided
    try:
        if bkg_slice_indices is not None:
            indices_out = os.path.join(
                results_dir, f"{tile_name}_bkg_slice_indices.json"
            )
            with open(indices_out, "w", encoding="utf-8") as handle:
                json.dump(
                    np.asarray(bkg_slice_indices, dtype=np.int64).tolist(),
                    handle,
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


def process_tile(
    tile_path: str,
    *,
    args: argparse.Namespace,
    method: str,
    res: str,
    binned_channel: str | None,
    binned_res: str,
    results_dir: str,
    out_path: str,
    out_mask_path: str,
    out_probability_path: str,
    data_process: Any,
    fitting_config: FittingConfig | None,
    median_summary: dict[str, float],
    mask_artifacts: MaskArtifacts | None,
) -> MaskArtifacts | None:
    """Execute the flatfield pipeline for a single tile and persist outputs."""

    tile_name = Path(tile_path).name
    _LOGGER.info(f"Processing tile: {tile_name}")
    report_path = os.path.join(results_dir, f"dask-report_{tile_name}.html")

    with performance_report(filename=report_path):
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
            bkg_slice_indices = None
            if not args.skip_bkg_sub:
                _LOGGER.info("Performing background subtraction")
                full_res, bkg, bkg_slice_indices = background_subtraction(
                    tile_path,
                    full_res,
                    z,
                    is_binned_channel,
                    args.use_reference_bkg,
                )

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
                    if bkg is None or bkg_slice_indices is None:
                        raise ValueError(
                            "Fitting method requires background subtraction. "
                            "Ensure background estimation is executed before fitting."
                        )
                    apply_median_summary_override(
                        fitting_config,
                        median_summary,
                        tile_name,
                        is_binned_channel=is_binned_channel,
                        binned_channel=binned_channel,
                    )
                    corrected, axis_fits, mask_artifacts = flatfield_fitting(
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
                        bkg=bkg,
                        bkg_slice_indices=bkg_slice_indices,
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
                write_empty_chunks=True,
            )
            _LOGGER.info(f"Storing OME-Zarr took {time.time() - t0:.2f}s")

            artifacts = None
            if method == "fitting":
                artifacts = {"axis_fits": axis_fits}

            save_method_outputs(
                method,
                tile_name,
                results_dir,
                args.save_outputs,
                bkg=bkg,
                bkg_slice_indices=bkg_slice_indices,
                artifacts=artifacts,
            )

            data_process.end_date_time = datetime.now()
            save_metadata(
                data_process, out_path, tile_name, tile_path, results_dir
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                f"Error processing tile {tile_name}: {exc}", exc_info=True
            )
            raise

    return mask_artifacts


def get_channel_resolution(
    tile_name: str, binned_channel: str | None, binned_res: str, res: str
) -> tuple[bool, str]:
    """
    Determine if the tile is a binned channel and select the appropriate
    zarr resolution.

    Parameters
    ----------
    tile_name : str
        Name of the tile being processed.
    binned_channel : str or None
        Name of the binned channel pattern, if available.
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
    parser.add_argument(
        "--worker-mode",
        type=str,
        choices=["processes", "threads"],
        default="processes",
        help="Execution mode for Dask workers (default: processes).",
    )
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
        default=None,
        help=(
            "Path to a JSON file containing per-channel mean_of_medians values "
            "for global normalization. Overrides the value stored in the fitting "
            "configuration when provided."
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


def create_zarr_artifact_path(out_zarr_path: str, dirname: str) -> str:
    """Derive the OME-Zarr artifact output path from the corrected dataset path.

    Parameters
    ----------
    out_zarr_path : str
        Path to the corrected OME-Zarr dataset.

    Returns
    -------
    str
        Path pointing to the associated artifact dataset.
    """
    out_zarr_folder = Path(out_zarr_path).name
    out_path = out_zarr_path.replace(
        out_zarr_folder, f"{dirname}/{out_zarr_folder}"
    )
    return out_path


def main() -> None:
    """Execute the flatfield correction pipeline for the requested tiles."""
    args = parse_and_validate_args()
    _LOGGER.setLevel(args.log_level)

    _LOGGER.info(f"args: {args}")

    params = resolve_args(args)
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

    client = create_dask_client(
        results_dir=results_dir,
        num_workers=args.num_workers,
        worker_mode=args.worker_mode,
    )
    _LOGGER.info(f"Dask client: {client}")

    out_mask_path = create_zarr_artifact_path(out_path, "mask")
    out_probability_path = create_zarr_artifact_path(out_path, "probability")
    initialize_zarr_group(out_mask_path)
    initialize_zarr_group(out_probability_path)

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
        if args.median_summary_path:
            fitting_config.median_summary_path = Path(args.median_summary_path)
        # dump to results folder
        fitting_config.to_file(
            os.path.join(
                results_dir, f"fitting_config_{tile_name_for_artifacts}.json"
            )
        )
        median_summary = read_median_intensity_summary(
            fitting_config.median_summary_path
        )
        if median_summary:
            _LOGGER.info(
                "Loaded median intensity overrides for channels: %s",
                ", ".join(sorted(median_summary)),
            )

    for tile_path in tile_paths:
        mask_artifacts = process_tile(
            tile_path,
            args=args,
            method=method,
            res=res,
            binned_channel=binned_channel,
            binned_res=binned_res,
            results_dir=results_dir,
            out_path=out_path,
            out_mask_path=out_mask_path,
            out_probability_path=out_probability_path,
            data_process=data_process,
            fitting_config=fitting_config,
            median_summary=median_summary,
            mask_artifacts=mask_artifacts,
        )

    if artifacts_destination:
        upload_artifacts(results_dir, artifacts_destination)


if __name__ == "__main__":
    main()
