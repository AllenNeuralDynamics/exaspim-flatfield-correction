import argparse
import json
import logging
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import dask.array as da
import tifffile
from dask.distributed import performance_report
from dask_image.ndfilters import gaussian_filter as gaussian_filter_dask
from scipy.ndimage import binary_fill_holes

from exaspim_flatfield_correction.background import (
    estimate_bkg,
    estimate_bkg_from_mapped_slices,
    subtract_bkg,
)
from exaspim_flatfield_correction.basic import fit_basic, transform_basic
from exaspim_flatfield_correction.config import (
    FittingConfig,
    PipelineConfig,
    apply_median_summary_override,
    load_fitting_config,
    load_pipeline_config,
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
    get_bkg_path,
    get_parent_s3_path,
    load_mask_from_dir,
    read_bkg_image,
    resize,
    save_correction_curve_plot,
    upload_artifacts,
    weighted_percentile,
)
from exaspim_flatfield_correction.utils.zarr_utils import (
    ensure_group,
    parse_ome_zarr_transformations,
    store_ome_zarr,
)
from exaspim_flatfield_correction.utils.dask_utils import create_dask_client
from xarray_multiscale.reducers import windowed_mode, windowed_rank
from zarr_io.arrays import open_group, read_zarr_array
from zarr_io.backends import (
    ArrayIOBackend,
    configure_io_backend_on_dask_workers,
    io_backend_from_name,
)
from zarr_io.config import IOBackendName


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
_LOGGER = logging.getLogger(__name__)


@contextmanager
def safe_performance_report(filename: str):
    """Wrap dask's ``performance_report`` so a report-rendering failure never
    aborts a completed run.

    The performance report is pure diagnostics, but it is generated in the
    context manager's ``__exit__`` -- i.e. *after* all tile work has finished.
    bokeh/distributed version skew (e.g. bokeh>=3.9 renaming ``file.html`` to
    ``file.html.jinja``, which distributed 2024.11.2 still extends) raises a
    ``TemplateNotFound`` there, which would otherwise discard hours of finished
    work. Any error writing the report is logged and swallowed; genuine errors
    from the wrapped body still propagate.
    """
    report = performance_report(filename=filename)
    report.__enter__()
    exc_info = (None, None, None)
    try:
        yield
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        try:
            report.__exit__(*exc_info)
        except Exception as report_exc:  # noqa: BLE001
            _LOGGER.warning(
                f"Failed to write dask performance report to {filename}: "
                f"{report_exc}"
            )


@dataclass
class MaskArtifacts:
    """Container for cached mask artifacts reused within the pipeline."""

    mask_low_res: da.Array
    probability_volume: da.Array | None = None


def _save_local_zarr(
    path: str, array: np.ndarray, chunks: tuple[int, ...]
) -> str:
    """Write a numpy array to a local Zarr array (overwriting).

    Used for ephemeral, local-only caches (background images, preprocessed
    masks). Uses the default Zarr v3 zstd compressor; reads use ``da.from_zarr``.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    z_arr = zarr.create_array(
        store=str(path),
        shape=tuple(int(s) for s in array.shape),
        dtype=array.dtype,
        chunks=tuple(int(c) for c in chunks),
    )
    z_arr[...] = array
    return str(path)


def resolve_output_format(args: PipelineConfig) -> int:
    """Resolve the output Zarr format (2 or 3) from config and the input tile.

    ``output_zarr_format="match"`` reads the first input tile's format so the
    corrected output round-trips in the same Zarr format as the input.
    """
    preference = args.output_zarr_format
    if preference in ("2", "3"):
        return int(preference)
    first_tile = open_group(args.tile_paths[0], mode="r")
    return int(first_tile.metadata.zarr_format)


def background_subtraction(
    tile_path: str,
    full_res: da.Array,
    z: zarr.Group,
    results_dir: str | None = None,
    tile_name: str | None = None,
    is_binned_channel: bool = False,
    use_reference_bkg: bool = False,
    background_smoothing_sigma: float = 1.0,
    background_final_smoothing_sigma: float = 0.0,
    target_resolution: str = "0",
    io_backend: IOBackendName | ArrayIOBackend = "tensorstore",
) -> tuple[da.Array, np.ndarray, np.ndarray | None, Path | None]:
    """Remove background estimates from the full-resolution volume.

    Parameters
    ----------
    tile_path : str
        Filesystem or S3 path to the tile Zarr group.
    full_res : dask.array.Array
        Full-resolution image volume to correct (modified lazily).
    z : zarr.Group
        Open Zarr hierarchy for the tile.
    results_dir : str or None, default=None
        Directory where the temporary background Zarr cache should be written.
    tile_name : str or None, default=None
        Tile identifier used to create unique Dask names and cache paths.
    is_binned_channel : bool, default=False
        Flag indicating whether the tile represents the binned channel.
    use_reference_bkg : bool, default=False
        When ``True`` load a static reference background, otherwise estimate
        it from the data.
    background_smoothing_sigma : float, default=1.0
        Gaussian smoothing sigma applied before background estimation. Set to
        0 to disable smoothing. Smoothing is only used to select background
        slices; estimated subtraction backgrounds are built from raw target
        resolution planes.
    background_final_smoothing_sigma : float, default=0.0
        Gaussian smoothing sigma applied to the final target-resolution 2D
        background image before subtraction and artifact saving. Set to 0 to
        disable final background smoothing.
    target_resolution : str, default="0"
        Resolution key of ``full_res`` within ``z``. Used to map background
        slice selections from the estimation resolution to the corrected
        target resolution.

    Returns
    -------
    tuple
        ``(full_res_corrected, background_image, background_slice_indices,
        background_cache_path)``. ``background_image`` is in the target
        resolution, while ``background_slice_indices`` remain in the
        estimation-resolution coordinate system. ``background_slice_indices``
        may be ``None`` when a reference background is used.
    """
    if results_dir is None:
        raise ValueError(
            "results_dir is required for background cache writes."
        )
    if tile_name is None:
        tile_name = Path(tile_path).name
    if background_final_smoothing_sigma < 0:
        raise ValueError(
            "background_final_smoothing_sigma must be non-negative."
        )

    dask_name_token = uuid.uuid4().hex
    safe_tile_name = _safe_dask_name(tile_name)

    if use_reference_bkg:
        bkg_path = get_bkg_path(tile_path)
        bkg = read_bkg_image(bkg_path).astype(np.float32)
        bkg_slice_indices = None
        if bkg.shape != full_res.shape[1:]:
            bkg = resize(bkg, full_res.shape[1:]).astype(
                np.float32, copy=False
            )
    else:
        bkg_res = "0" if is_binned_channel else "3"
        _LOGGER.info(f"Using resolution {bkg_res} for background estimation")
        _LOGGER.info(
            "Background estimation smoothing sigma: %s",
            background_smoothing_sigma,
        )
        low_res = read_zarr_array(
            z, component=bkg_res, io_backend=io_backend
        ).squeeze().astype(np.float32)
        low_res_shape = tuple(int(dim) for dim in low_res.shape)
        if background_smoothing_sigma > 0:
            low_res = gaussian_filter_dask(
                low_res, sigma=background_smoothing_sigma
            )
        _, bkg_slice_indices = estimate_bkg(low_res.compute())

        source_transformations = parse_ome_zarr_transformations(z, bkg_res)
        target_transformations = parse_ome_zarr_transformations(
            z, target_resolution
        )
        bkg, target_slice_indices = estimate_bkg_from_mapped_slices(
            full_res,
            bkg_slice_indices,
            low_res_shape,
            source_transformations=source_transformations,
            target_transformations=target_transformations,
        )
        _LOGGER.info(
            "Computed target-resolution background from target slices %s..%s",
            int(target_slice_indices[0]),
            int(target_slice_indices[-1]),
        )

    _LOGGER.info(
        "Final background smoothing sigma: %s",
        background_final_smoothing_sigma,
    )
    cache_dir = _background_cache_dir(results_dir, tile_name, dask_name_token)
    spatial_chunks = chunks_2d(full_res)
    try:
        if background_final_smoothing_sigma > 0:
            raw_cache_path = _write_background_cache(
                bkg,
                cache_dir / "raw.zarr",
                spatial_chunks,
            )
            raw_bkg_da = _read_background_cache(
                raw_cache_path,
                name=f"bkg-cache-raw-{safe_tile_name}-{dask_name_token}",
            )
            blurred_bkg_da = gaussian_filter_dask(
                raw_bkg_da,
                sigma=background_final_smoothing_sigma,
            ).astype(np.float32)
            bkg = blurred_bkg_da.compute().astype(np.float32, copy=False)

        final_cache_path = _write_background_cache(
            bkg,
            cache_dir / "final.zarr",
            spatial_chunks,
        )
        bkg_da = _read_background_cache(
            final_cache_path,
            name=f"bkg-cache-final-{safe_tile_name}-{dask_name_token}",
        )
    except Exception:
        cleanup_background_cache(cache_dir)
        raise

    full_res = subtract_bkg(
        full_res,
        bkg_da,
    )
    return full_res, bkg, bkg_slice_indices, cache_dir


def _safe_dask_name(value: str) -> str:
    """Normalize user-facing identifiers for Dask graph layer names."""

    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )


def _background_cache_dir(
    results_dir: str,
    tile_name: str,
    token: str,
) -> Path:
    """Create a unique local cache directory for a tile background image."""

    cache_root = Path(results_dir) / "dask-temp" / "background-cache"
    cache_dir = cache_root / f"{_safe_dask_name(tile_name)}_{token}"
    cache_dir.mkdir(parents=True, exist_ok=False)
    return cache_dir


def _write_background_cache(
    bkg: np.ndarray,
    cache_path: Path,
    chunks: tuple[int, ...],
) -> Path:
    """Write a 2D background image to a local chunked Zarr array."""

    _save_local_zarr(str(cache_path), np.asarray(bkg, dtype=np.float32), chunks)
    return cache_path


def _read_background_cache(cache_path: Path, name: str) -> da.Array:
    """Read a cached background image with a unique Dask graph name."""

    return da.from_zarr(str(cache_path), name=name).astype(np.float32)


def cleanup_background_cache(cache_path: Path | None) -> None:
    """Remove a temporary background cache directory if it exists."""

    if cache_path is not None:
        shutil.rmtree(cache_path, ignore_errors=True)


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
    z: zarr.Group,
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
    io_backend: IOBackendName | ArrayIOBackend = "tensorstore",
) -> da.Array:
    """Apply BasicPy flatfield correction to the supplied volume.

    Parameters
    ----------
    full_res : dask.array.Array
        Full-resolution image volume to correct.
    z : zarr.Group
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
    low_res = read_zarr_array(
        z, component=basicpy_res, io_backend=io_backend
    ).squeeze().astype(np.float32)
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
    _save_local_zarr(str(mask_name), mask, chunks)
    return da.from_zarr(mask_name)


def _create_mask_artifacts(
    low_res: da.Array,
    z: zarr.Group,
    fitting_res: str,
    mask_dir: str,
    tile_name: str,
    results_dir: str,
    config: FittingConfig,
    bkg_slice_indices: np.ndarray,
    out_probability_path: str,
    overwrite: bool,
    output_zarr_format: int = 2,
    io_backend: IOBackendName | ArrayIOBackend = "tensorstore",
    corrected_rank: int = -2,
    max_chunks_per_block: int | None = 16384,
) -> MaskArtifacts:
    """Generate mask artifacts and optional probability volumes for a tile.

    Parameters
    ----------
    low_res : dask.array.Array
        Low-resolution stack used for mask inference.
    z : zarr.Group
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
            zarr_format=output_zarr_format,
            io_backend=io_backend,
            reducer=partial(windowed_rank, rank=corrected_rank),
            write_empty_chunks=False,
            max_chunks_per_block=max_chunks_per_block,
        )
        # Do not materialize into memory until needed
        probability_volume = read_zarr_array(
            probability_path, component="0", io_backend=io_backend
        ).squeeze()

    return MaskArtifacts(
        mask_low_res=initial_mask,
        probability_volume=probability_volume,
    )


def flatfield_fitting(
    full_res: da.Array,
    z: zarr.Group,
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
    output_zarr_format: int = 2,
    io_backend: IOBackendName | ArrayIOBackend = "tensorstore",
    corrected_rank: int = -2,
    max_chunks_per_block: int | None = 16384,
) -> tuple[da.Array, dict[str, np.ndarray], MaskArtifacts | None]:
    """Run the fitting-based flatfield workflow for a single tile.

    Parameters
    ----------
    full_res : dask.array.Array
        Full-resolution volume to correct.
    z : zarr.Group
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
    low_res = read_zarr_array(
        z, component=fitting_res, io_backend=io_backend
    ).squeeze().astype(np.float32)

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
            output_zarr_format=output_zarr_format,
            io_backend=io_backend,
            corrected_rank=corrected_rank,
            max_chunks_per_block=max_chunks_per_block,
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
        zarr_format=output_zarr_format,
        io_backend=io_backend,
        reducer=windowed_mode,
        write_empty_chunks=False,
        max_chunks_per_block=max_chunks_per_block,
    )
    # Re-read the computed mask back from S3 for performance
    mask_upscaled = read_zarr_array(
        mask_path, component="0", io_backend=io_backend
    ).squeeze()

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
        low_res.round().astype(np.uint16),
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
    low_res = da.clip(low_res, 0, global_val * med_factor)
    _LOGGER.info(f"Smoothing low res with sigma: {profile_sigma}")
    low_res = gaussian_filter_dask(low_res, sigma=profile_sigma).compute()

    _LOGGER.info(
        "Computing masked profiles (sigma=%s, percentile=%s, min_voxels=%s)",
        profile_sigma,
        profile_percentile,
        profile_min_voxels,
    )
    axis_fits = compute_axis_fits(
        low_res,
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

    # Save background-derived outputs independently when provided.
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
        if bkg is not None:
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
    args: PipelineConfig,
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
    output_format: int,
) -> MaskArtifacts | None:
    """Execute the flatfield pipeline for a single tile and persist outputs."""

    tile_name = Path(tile_path).name
    _LOGGER.info(f"Processing tile: {tile_name}")
    # Build the I/O backend once with the configured concurrency limits baked in.
    # Threading the backend object (rather than the backend name) makes both reads
    # and writes honor IOConcurrencyConfig, including across pickled Dask tasks.
    io_backend = io_backend_from_name(
        args.io_backend, args.io_concurrency.to_io_concurrency()
    )
    report_path = os.path.join(results_dir, f"dask-report_{tile_name}.html")
    background_cache_path: Path | None = None

    with safe_performance_report(filename=report_path):
        try:
            if args.is_binned:
                is_binned_channel = True
                resolution = "0"
            else:
                is_binned_channel, resolution = get_channel_resolution(
                    tile_name, binned_channel, binned_res, res
                )
            _LOGGER.info(f"{tile_name} is binned: {is_binned_channel}")

            z = open_group(tile_path, mode="r")
            coordinate_transformations = parse_ome_zarr_transformations(
                z, resolution
            )
            _LOGGER.info(
                f"Coordinate transformations: {coordinate_transformations}"
            )

            full_res = read_zarr_array(
                z, component=resolution, io_backend=io_backend
            ).squeeze().astype(np.float32)
            _LOGGER.info(f"Full resolution array shape: {full_res.shape}")

            bkg = None
            bkg_slice_indices = None
            if not args.skip_bkg_sub:
                _LOGGER.info("Performing background subtraction")
                (
                    full_res,
                    bkg,
                    bkg_slice_indices,
                    background_cache_path,
                ) = background_subtraction(
                    tile_path,
                    full_res,
                    z,
                    results_dir,
                    tile_name,
                    is_binned_channel,
                    use_reference_bkg=args.use_reference_bkg,
                    background_smoothing_sigma=args.background_smoothing_sigma,
                    background_final_smoothing_sigma=(
                        args.background_final_smoothing_sigma
                    ),
                    target_resolution=resolution,
                    io_backend=io_backend,
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
                        io_backend=io_backend,
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
                        output_zarr_format=output_format,
                        io_backend=io_backend,
                        corrected_rank=args.corrected_rank,
                        max_chunks_per_block=args.max_chunks_per_block,
                    )
                else:
                    _LOGGER.error(f"Invalid method: {method}")
                    raise ValueError(f"Invalid method: {method}")
            else:
                corrected = full_res

            corrected = corrected.round().astype(np.uint16)
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
                zarr_format=output_format,
                io_backend=io_backend,
                reducer=partial(windowed_rank, rank=args.corrected_rank),
                write_empty_chunks=True,
                max_chunks_per_block=args.max_chunks_per_block,
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
            save_metadata(data_process, tile_name, results_dir)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                f"Error processing tile {tile_name}: {exc}", exc_info=True
            )
            raise
        finally:
            cleanup_background_cache(background_cache_path)

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


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the flatfield correction pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config file containing pipeline inputs.",
    )
    return parser


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
    parser = build_parser()
    cli_args = parser.parse_args()
    args = load_pipeline_config(cli_args.config)
    _LOGGER.setLevel(args.log_level)

    _LOGGER.info("Pipeline parameters: %s", args.model_dump(mode="json"))

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

    # Propagate Zarr concurrency limits to every worker process. This is a no-op
    # for the tensorstore backend (whose limits travel with the pickled backend
    # object) but is required for the zarr backend, whose zarr.config must be set
    # on the workers themselves rather than only in the client process.
    configure_io_backend_on_dask_workers(
        client, args.io_backend, args.io_concurrency.to_io_concurrency()
    )

    output_format = resolve_output_format(args)
    _LOGGER.info("Output Zarr format: v%d", output_format)

    out_mask_path = create_zarr_artifact_path(args.output, "mask")
    out_probability_path = create_zarr_artifact_path(args.output, "probability")
    ensure_group(out_mask_path, zarr_format=output_format)
    ensure_group(out_probability_path, zarr_format=output_format)

    artifacts_destination = None
    tile_name_for_artifacts = (
        Path(args.tile_paths[0]).name if args.tile_paths else None
    )
    if tile_name_for_artifacts and args.output.startswith("s3://"):
        try:
            parent_output = get_parent_s3_path(args.output)
            artifacts_destination = (
                f"{parent_output}/artifacts/{tile_name_for_artifacts}"
            )
        except ValueError:
            _LOGGER.warning(
                "Unable to determine artifacts destination from output path %s",
                args.output,
            )

    start_date_time = datetime.now()
    data_process = create_processing_metadata(
        args, args.tile_paths[0], args.output, start_date_time, args.res
    )

    mask_artifacts: MaskArtifacts | None = None
    fitting_config: FittingConfig | None = None
    median_summary: dict[str, float] = {}
    if args.method == "fitting":
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

    for tile_path in args.tile_paths:
        mask_artifacts = process_tile(
            tile_path,
            args=args,
            method=args.method,
            res=args.res,
            binned_channel=args.binned_channel,
            binned_res=binned_res,
            results_dir=results_dir,
            out_path=args.output,
            out_mask_path=out_mask_path,
            out_probability_path=out_probability_path,
            data_process=data_process,
            fitting_config=fitting_config,
            median_summary=median_summary,
            mask_artifacts=mask_artifacts,
            output_format=output_format,
        )

    if artifacts_destination:
        upload_artifacts(results_dir, artifacts_destination)


if __name__ == "__main__":
    main()
