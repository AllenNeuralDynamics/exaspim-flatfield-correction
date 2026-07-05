"""Tests for choosing the output Zarr format independently of the input.

Covers ``resolve_output_format`` (the config -> int resolution, including the
"match" path that reads the input tile's format) and a true cross-format
read->write round-trip (v3 input -> v2 output and vice versa) through the same
``zarr-io`` helpers the pipeline uses.

All paths are local ``tmp_path`` and the zarr backend is used throughout, so the
tests require neither tensorstore nor S3.
"""

from __future__ import annotations

import numpy as np
import pytest

import dask.array as da

from exaspim_flatfield_correction.config import PipelineConfig
from exaspim_flatfield_correction.pipeline import resolve_output_format
from exaspim_flatfield_correction.utils.zarr_utils import (
    ensure_group,
    parse_ome_zarr_transformations,
    store_ome_zarr,
)
from zarr_io.arrays import open_group, read_zarr_array


def _make_input_tile(
    path: str,
    *,
    zarr_format: int,
    raw: np.ndarray,
    voxel_size: tuple[float, float, float],
    origin: tuple[float, ...],
) -> None:
    """Write a real input OME-Zarr tile (data + metadata) in ``zarr_format``."""
    data = da.from_array(raw, chunks=raw.shape)
    store_ome_zarr(
        data,
        path,
        n_levels=1,
        voxel_size=voxel_size,
        origin=origin,
        overwrite=True,
        zarr_format=zarr_format,
        io_backend="zarr",
    )


def _pipeline_config(tile_path: str, output_zarr_format: str) -> PipelineConfig:
    """Minimal valid config for exercising ``resolve_output_format``.

    ``method="basicpy"`` avoids the fitting cross-field requirements
    (mask_dir / background subtraction).
    """
    return PipelineConfig.model_validate(
        {
            "method": "basicpy",
            "tile_paths": [str(tile_path)],
            "output": "/tmp/flatfield_out.ome.zarr",
            "output_zarr_format": output_zarr_format,
        }
    )


@pytest.mark.parametrize("in_fmt", [2, 3])
def test_resolve_output_format_match_follows_input(tmp_path, in_fmt) -> None:
    tile_path = str(tmp_path / f"tile_v{in_fmt}.zarr")
    ensure_group(tile_path, zarr_format=in_fmt, mode="w")

    cfg = _pipeline_config(tile_path, "match")

    assert resolve_output_format(cfg) == in_fmt


@pytest.mark.parametrize(
    "in_fmt, forced",
    [(3, "2"), (2, "3"), (2, "2"), (3, "3")],
)
def test_resolve_output_format_forced_overrides_input(
    tmp_path, in_fmt, forced
) -> None:
    tile_path = str(tmp_path / f"tile_v{in_fmt}.zarr")
    ensure_group(tile_path, zarr_format=in_fmt, mode="w")

    cfg = _pipeline_config(tile_path, forced)

    # The forced format wins regardless of the input tile's format.
    assert resolve_output_format(cfg) == int(forced)


@pytest.mark.parametrize(
    "in_fmt, out_fmt",
    [(3, 2), (2, 3), (2, 2), (3, 3)],
)
def test_cross_format_conversion_roundtrip(tmp_path, in_fmt, out_fmt) -> None:
    """Read a tile written in ``in_fmt`` and rewrite it in ``out_fmt``.

    Exercises the pipeline's actual read->write boundary (open_group +
    read_zarr_array + parse_ome_zarr_transformations -> store_ome_zarr).
    """
    raw = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    voxel_size = (2.0, 0.5, 0.5)
    origin = (0, 0, 10, 20, 30)

    in_path = str(tmp_path / f"in_v{in_fmt}.zarr")
    out_path = str(tmp_path / f"out_v{out_fmt}.zarr")
    _make_input_tile(
        in_path, zarr_format=in_fmt, raw=raw, voxel_size=voxel_size, origin=origin
    )

    # Read exactly as the pipeline does (format-agnostic).
    z = open_group(in_path, mode="r")
    assert int(z.metadata.zarr_format) == in_fmt
    data = read_zarr_array(z, component="0", io_backend="zarr")
    transforms = parse_ome_zarr_transformations(z, "0")

    # Write in the (possibly different) output format.
    store_ome_zarr(
        data,
        out_path,
        n_levels=1,
        voxel_size=tuple(transforms["scale"][-3:]),
        origin=transforms["translation"],
        overwrite=True,
        zarr_format=out_fmt,
        io_backend="zarr",
    )

    out = open_group(out_path, mode="r")
    assert int(out.metadata.zarr_format) == out_fmt

    out_data = (
        read_zarr_array(out, component="0", io_backend="zarr").squeeze().compute()
    )
    np.testing.assert_array_equal(out_data.astype(np.uint16), raw)

    # Scale/translation survive the cross-format conversion.
    out_transforms = parse_ome_zarr_transformations(out, "0")
    assert out_transforms["scale"] == [1.0, 1.0, 2.0, 0.5, 0.5]
    assert out_transforms["translation"] == [0.0, 0.0, 10.0, 20.0, 30.0]
