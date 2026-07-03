"""Zarr I/O and OME-Zarr multiscale writing for the flatfield pipeline.

Reads and writes both Zarr v2 and Zarr v3 (OME-NGFF 0.4 / 0.5) by delegating to
the ``zarr-io`` and ``zarr-multiscale`` libraries. The corrected output is written
in the same Zarr format as the input tile (chosen by the caller).
"""

import logging
from functools import partial
from typing import Any, Callable

import dask.array as da
import numpy as np
import zarr
from zarr.errors import ContainsGroupError

from xarray_multiscale import multiscale as _xarray_multiscale
from xarray_multiscale.reducers import windowed_rank

from zarr_io.arrays import (
    ArraySpec,
    ArrayTarget,
    open_group,
    read_zarr_array,
    write_dask_array,
)
from zarr_io.backends import ArrayIOBackend, io_backend_from_name
from zarr_io.config import IOBackendName, IOConcurrencyConfig, OutputShards
from zarr_io.ome import (
    axes_from_attrs,
    parse_ome_zarr_transformations,  # re-exported for callers (v0.4 + v0.5 aware)
    write_ome_ngff_metadata,
)
from zarr_multiscale.pyramid import reducer_name, write_multiscale_pyramid

_LOGGER = logging.getLogger(__name__)

DEFAULT_SCALE_FACTORS = (1, 1, 2, 2, 2)
DEFAULT_REDUCER = partial(windowed_rank, rank=-2)

__all__ = [
    "store_ome_zarr",
    "parse_ome_zarr_transformations",
    "ensure_group",
    "get_zarr_tiles",
]


def ensure_group(
    path: str,
    *,
    zarr_format: int,
    mode: str = "a",
    aws_region: str = "us-west-2",
    s3_use_ssl: bool = False,
    s3_total_max_attempts: int = 10,
    s3_retry_mode: str = "adaptive",
) -> zarr.Group:
    """Open (creating if needed) a container Zarr group, local or on S3.

    Replaces the old ``initialize_zarr_group`` helper; supports both Zarr v2 and
    v3 via ``zarr-io``.
    """
    return open_group(
        path,
        mode=mode,
        zarr_format=zarr_format,
        aws_region=aws_region,
        s3_use_ssl=s3_use_ssl,
        s3_total_max_attempts=s3_total_max_attempts,
        s3_retry_mode=s3_retry_mode,
    )


def store_ome_zarr(
    data: da.Array,
    output_zarr: str,
    n_levels: int = 1,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, ...] = (0, 0, 0, 0, 0),
    *,
    overwrite: bool = False,
    zarr_format: int = 2,
    io_backend: IOBackendName | ArrayIOBackend = "tensorstore",
    io_concurrency: IOConcurrencyConfig | None = None,
    reducer: Callable[..., Any] = DEFAULT_REDUCER,
    write_empty_chunks: bool = True,
    max_chunks_per_block: int | None = 16384,
    scale_factors: tuple[int, ...] | None = None,
    chunks: tuple[int, ...] | None = None,
    output_shards: OutputShards = "none",
    codec: Any | None = None,
    aws_region: str = "us-west-2",
    s3_use_ssl: bool = False,
    s3_total_max_attempts: int = 10,
    s3_retry_mode: str = "adaptive",
) -> None:
    """Store a Dask array as an OME-Zarr multiscale dataset (Zarr v2 or v3).

    Writes level ``0`` directly, then generates pyramid levels ``1..n-1`` with
    ``zarr-multiscale`` and writes OME-NGFF metadata. The output Zarr format is
    chosen by ``zarr_format`` (match the input tile for round-trip compatibility).

    Parameters
    ----------
    data : dask.array.Array
        Image data to store (expanded to 5D TCZYX if needed).
    output_zarr : str
        Local path or ``s3://`` URI of the output OME-Zarr group.
    n_levels : int
        Requested number of pyramid levels (clamped to what the shape supports).
    voxel_size : tuple of float
        Physical voxel size (ZYX) written as the level-0 scale transform.
    origin : tuple of float
        Translation transform (TCZYX, or ZYX which is front-padded with zeros).
    overwrite : bool
        Overwrite an existing group; when False an existing group is left intact
        (the tile is skipped).
    zarr_format : int
        Output Zarr format, 2 or 3.
    io_backend : str or ArrayIOBackend
        ``"tensorstore"`` (default) or ``"zarr"``. Both honor
        ``write_empty_chunks``.
    reducer : Callable
        Downsampling reducer (e.g. ``partial(windowed_rank, rank=-2)`` for image
        data, ``windowed_mode`` for label/mask data).
    write_empty_chunks : bool
        When False, chunks equal to the fill value are not written (sparse output).
    max_chunks_per_block : int or None
        Maximum chunks per ``dask.compute`` for the level-0 write. Large arrays
        are written in chunk-aligned slabs of at most this many chunks so the
        scheduler isn't overwhelmed by one giant graph. ``None`` writes in a
        single compute (zarr_io's legacy behavior).
    scale_factors : tuple of int, optional
        Per-axis (TCZYX) downsampling factors. Defaults to (1, 1, 2, 2, 2). A
        3-tuple (ZYX) is front-padded with (1, 1).
    chunks : tuple of int, optional
        Level-0 chunk shape (TCZYX). Defaults to the source array's own chunking
        (``data.chunksize``).
    output_shards : "inherit" | "none" | None | tuple
        Shard layout for written levels (Zarr v3 only); "none" (or None)
        disables sharding.
    codec : optional
        Compressor. Defaults to zstd Blosc appropriate to ``zarr_format``.
    """
    if zarr_format not in (2, 3):
        raise ValueError(f"Unsupported Zarr format: {zarr_format}")

    # Accept None as an alias for "none": zarr_io's parse_output_shards only
    # understands the string sentinels ("inherit"/"none") or a shard tuple, and
    # chokes on None inside write_multiscale_pyramid.
    if output_shards is None:
        output_shards = "none"

    if scale_factors is None:
        scale_factors = DEFAULT_SCALE_FACTORS
    if len(scale_factors) == 3:
        scale_factors = (1, 1, *scale_factors)
    if len(scale_factors) != 5:
        raise ValueError("scale_factors must describe all five TCZYX axes")

    # Expand to 5D TCZYX.
    while data.ndim < 5:
        data = data[np.newaxis, ...]

    # Default the chunk shape to the source array's own chunking (TCZYX). Inheriting
    # the input chunks keeps the output layout aligned with the tile being written.
    if chunks is None:
        chunks = data.chunksize
    if len(chunks) != 5:
        raise ValueError("chunks must be a tuple of length 5")

    # Clamp the chunk shape to the array shape so a single chunk never exceeds
    # the data; otherwise small volumes try to allocate a full oversized chunk.
    chunks = tuple(
        max(1, min(int(c), int(s))) for c, s in zip(chunks, data.shape)
    )

    # Normalize an explicit shard request (Zarr v3). A 3-tuple is ZYX and is
    # front-padded to TCZYX. Zarr v3 requires the shard shape to be a whole
    # multiple of the inner chunk along every axis, and neither zarr-io nor
    # zarr-multiscale enforces that, so snap each axis to the nearest chunk
    # multiple (at least one chunk). The same shard is applied to every pyramid
    # level; a shard larger than a level's shape is fine (one partial shard).
    if zarr_format == 3 and isinstance(output_shards, tuple):
        if len(output_shards) == 3:
            output_shards = (1, 1, *output_shards)
        if len(output_shards) != 5:
            raise ValueError("output_shards must describe all five TCZYX axes")
        output_shards = tuple(
            max(c, max(1, round(s / c)) * c)
            for s, c in zip(output_shards, chunks)
        )

    backend = io_backend_from_name(io_backend, io_concurrency)

    if codec is None:
        codec = _default_codec(zarr_format)

    target = _open_output_group(
        output_zarr,
        overwrite=overwrite,
        zarr_format=zarr_format,
        aws_region=aws_region,
        s3_use_ssl=s3_use_ssl,
        s3_total_max_attempts=s3_total_max_attempts,
        s3_retry_mode=s3_retry_mode,
    )
    if target is None:
        _LOGGER.info("Not overwriting existing tile %s", output_zarr)
        return

    # Level-0 spec (format-aware codec/shards).
    level0_shards = output_shards if (zarr_format == 3 and isinstance(output_shards, tuple)) else None
    spec_kwargs: dict[str, Any] = {
        "zarr_format": zarr_format,
        "chunks": tuple(chunks),
        "write_empty_chunks": write_empty_chunks,
        # An explicit (non-null) fill value is required for the tensorstore
        # backend to elide all-fill-value chunks when write_empty_chunks=False;
        # a null v2 fill value leaves nothing to compare against. 0 is the
        # background value for these volumes and matches the v3 default.
        "fill_value": 0,
    }
    if zarr_format == 2:
        spec_kwargs["compressor"] = codec
        # zarr-python defaults v2 arrays to the "." dimension separator; force
        # "/" so chunks are nested (matching the AIND OME-Zarr convention and
        # the pre-refactor output). The pyramid writer inherits this from the
        # level-0 array via ArraySpec.from_zarr.
        spec_kwargs["dimension_separator"] = "/"
    else:
        spec_kwargs["compressors"] = (codec,)
        spec_kwargs["shards"] = level0_shards
        spec_kwargs["dimension_names"] = ("t", "c", "z", "y", "x")
    spec0 = ArraySpec.from_dask(data, **spec_kwargs)

    _LOGGER.info("Writing level 0 to %s (zarr v%d, %s)", output_zarr, zarr_format, backend.name)
    # Slab the (potentially massive) level-0 write so the scheduler isn't handed a
    # single million-task graph. Pyramid levels are downsampled and far smaller;
    # write_multiscale_pyramid relies on zarr_io's own default budget for those.
    write_dask_array(
        data,
        target,
        "0",
        spec=spec0,
        io_backend=backend,
        overwrite=True,
        max_chunks_per_block=max_chunks_per_block,
    )

    # Generate pyramid levels 1..levels-1 (clamped to what the shape supports).
    levels = _max_levels(data, scale_factors, reducer, n_levels)
    if levels < n_levels:
        _LOGGER.warning(
            "Requested %d levels but shape %s only supports %d; clamping.",
            n_levels,
            tuple(int(s) for s in data.shape),
            levels,
        )
    if levels > 1:
        _LOGGER.info("Generating %d pyramid levels", levels - 1)
        write_multiscale_pyramid(
            target,
            target,
            io_backend=backend,
            n_levels=levels,
            scale_factors=scale_factors,
            output_shards=output_shards,
            reducer=reducer,
            ome_metadata=False,
            include_level_zero=True,
            write_empty_chunks=write_empty_chunks,
            max_chunks_per_block=max_chunks_per_block,
        )

    # Write OME-NGFF metadata, preserving voxel_size/origin from the input tile.
    base_scale = _to_5d_vector(voxel_size, fill=1.0)
    base_translation = _to_5d_vector(origin, fill=0.0)
    _LOGGER.info("Writing OME-NGFF metadata for %s", output_zarr)
    write_ome_ngff_metadata(
        target.group,
        source_attrs={},
        source_path=output_zarr,
        level_paths=[str(i) for i in range(levels)],
        base_scale=base_scale,
        base_translation=base_translation,
        scale_factors=scale_factors,
        axes=axes_from_attrs({}, ndim=5),
        downsample_type=reducer_name(reducer),
    )


def _open_output_group(
    output_zarr: str,
    *,
    overwrite: bool,
    zarr_format: int,
    **s3_kwargs: Any,
) -> ArrayTarget | None:
    """Open the output group, returning None if it exists and overwrite is False."""
    mode = "w" if overwrite else "w-"
    try:
        group = open_group(output_zarr, mode=mode, zarr_format=zarr_format, **s3_kwargs)
    except (FileExistsError, ContainsGroupError):
        return None
    return ArrayTarget.from_group(group, path=output_zarr)


def _default_codec(zarr_format: int) -> Any:
    """Return a zstd Blosc compressor appropriate to the Zarr format."""
    if zarr_format == 2:
        from numcodecs import blosc

        return blosc.Blosc(cname="zstd", clevel=1, shuffle=blosc.SHUFFLE)
    from zarr.codecs import BloscCodec, BloscShuffle

    return BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.shuffle)


def _to_5d_vector(values: tuple[float, ...] | list[float], *, fill: float) -> list[float]:
    """Coerce a transform vector to length 5 (TCZYX), front-padding with ``fill``."""
    vec = [float(v) for v in values]
    if len(vec) >= 5:
        return vec[:5]
    return [fill] * (5 - len(vec)) + vec


def _max_levels(
    data: da.Array,
    scale_factors: tuple[int, ...],
    reducer: Callable[..., Any],
    requested: int,
) -> int:
    """Number of pyramid levels (incl. level 0) the data supports, capped at ``requested``.

    Uses xarray-multiscale to count levels so the result matches exactly what
    ``write_multiscale_pyramid`` validates against (a heuristic can over-count and
    trigger a ValueError downstream). The lazy pyramid built here is discarded.
    """
    levels = _xarray_multiscale(data, reducer, scale_factors)
    return min(requested, len(levels))


def get_zarr_tiles(
    z: zarr.Group, res: int = 5, chunks: tuple[int, ...] = None
) -> list[da.Array]:
    """Extract tiles from a zarr group at a given resolution as Dask arrays.

    Parameters
    ----------
    z : zarr.Group
        Zarr group containing tiles.
    res : int, optional
        Resolution level to extract. Default is 5.
    chunks : tuple of int, optional
        Chunk size for the Dask arrays.
        If None, use the dataset's default chunks.

    Returns
    -------
    list of dask.array.Array
        List of Dask arrays for each tile at the specified resolution.
    """
    tiles = []
    for tile in sorted(z.keys()):
        ds = z[tile][res]
        if chunks is None:
            chunks = ds.chunks
        data = da.from_array(ds, chunks=chunks).squeeze()
        tiles.append(data)
    return tiles
