"""Tests for OME-Zarr writing (v2 + v3) via zarr-io / zarr-multiscale.

These exercise the rewritten ``store_ome_zarr`` and the v0.4/v0.5-aware
metadata parser. They require the new stack (zarr 3.x, zarr-io, zarr-multiscale,
xarray-multiscale); the TensorStore cases skip when tensorstore is unavailable.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from exaspim_flatfield_correction.utils.zarr_utils import (
    parse_ome_zarr_transformations,
    store_ome_zarr,
)
from zarr_io.arrays import open_group, read_zarr_array

BACKENDS = ["zarr", "tensorstore"]


def _maybe_skip_backend(io_backend: str) -> None:
    if io_backend == "tensorstore":
        pytest.importorskip("tensorstore")


def _multiscales(group, zarr_format: int) -> dict:
    attrs = dict(group.attrs)
    if zarr_format == 3:
        return attrs["ome"]["multiscales"][0]
    return attrs["multiscales"][0]


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("io_backend", BACKENDS)
def test_store_ome_zarr_roundtrip(tmp_path, zarr_format, io_backend) -> None:
    _maybe_skip_backend(io_backend)
    raw = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    data = da.from_array(raw, chunks=(8, 16, 16))
    out = str(tmp_path / f"tile_v{zarr_format}_{io_backend}.zarr")

    store_ome_zarr(
        data,
        out,
        n_levels=3,
        voxel_size=(2.0, 0.5, 0.5),
        origin=(0, 0, 10, 20, 30),
        overwrite=True,
        zarr_format=zarr_format,
        io_backend=io_backend,
        chunks=(1, 1, 4, 8, 8),
    )

    group = open_group(out, mode="r")
    assert int(group.metadata.zarr_format) == zarr_format
    for level in ("0", "1", "2"):
        assert level in group

    level0 = read_zarr_array(group, component="0", io_backend="zarr")
    assert tuple(int(s) for s in level0.shape) == (1, 1, 8, 16, 16)
    np.testing.assert_array_equal(level0.squeeze().compute().astype(np.uint16), raw)

    # Level-0 scale == [1, 1, *voxel_size]; translation preserved; level-1 doubled.
    t0 = parse_ome_zarr_transformations(group, "0")
    assert t0["scale"] == [1.0, 1.0, 2.0, 0.5, 0.5]
    assert t0["translation"] == [0.0, 0.0, 10.0, 20.0, 30.0]
    t1 = parse_ome_zarr_transformations(group, "1")
    assert t1["scale"] == [1.0, 1.0, 4.0, 1.0, 1.0]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_store_ome_zarr_inherits_source_chunks(tmp_path, zarr_format) -> None:
    """When ``chunks`` is omitted, the level-0 chunk shape inherits ``data.chunksize``."""
    raw = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    data = da.from_array(raw, chunks=(4, 8, 8))
    out = str(tmp_path / f"inherit_v{zarr_format}.zarr")

    store_ome_zarr(
        data,
        out,
        n_levels=1,
        overwrite=True,
        zarr_format=zarr_format,
        io_backend="zarr",
        # chunks intentionally omitted -> inherit from the source array.
    )

    group = open_group(out, mode="r")
    # 3D source expanded to 5D TCZYX: chunksize (4, 8, 8) -> (1, 1, 4, 8, 8).
    assert tuple(int(c) for c in group["0"].chunks) == (1, 1, 4, 8, 8)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_store_ome_zarr_mask_uses_windowed_mode(tmp_path, zarr_format) -> None:
    from xarray_multiscale.reducers import windowed_mode

    raw = (np.arange(4 * 8 * 8) % 2).astype(np.uint8).reshape(4, 8, 8)
    mask = da.from_array(raw, chunks=(4, 8, 8))
    out = str(tmp_path / f"mask_v{zarr_format}.zarr")

    store_ome_zarr(
        mask,
        out,
        n_levels=2,
        voxel_size=(1.0, 1.0, 1.0),
        origin=(0, 0, 0, 0, 0),
        overwrite=True,
        zarr_format=zarr_format,
        io_backend="zarr",
        reducer=windowed_mode,
        write_empty_chunks=False,
        chunks=(1, 1, 4, 8, 8),
    )

    group = open_group(out, mode="r")
    multiscale = _multiscales(group, zarr_format)
    assert multiscale["type"] == "windowed_mode"
    assert "1" in group


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_store_ome_zarr_tensorstore_skips_empty_chunks(tmp_path, zarr_format) -> None:
    """The tensorstore backend honors write_empty_chunks=False (sparse output).

    Regression guard for removing the old workaround that forced the zarr backend
    for sparse writes; requires zarr-io's tensorstore backend to map
    write_empty_chunks -> store_data_equal_to_fill_value.
    """
    pytest.importorskip("tensorstore")
    # All-zeros volume: every chunk equals the default (0) fill value.
    data = da.from_array(np.zeros((8, 16, 16), dtype=np.uint16), chunks=(4, 8, 8))
    out = str(tmp_path / f"sparse_v{zarr_format}.zarr")

    store_ome_zarr(
        data,
        out,
        n_levels=1,
        overwrite=True,
        zarr_format=zarr_format,
        io_backend="tensorstore",
        write_empty_chunks=False,
        chunks=(1, 1, 4, 8, 8),
    )

    # No chunk data files written for level 0 -- only metadata.
    metadata_names = {"zarr.json", ".zarray", ".zattrs", ".zgroup"}
    chunk_files = [
        p
        for p in (Path(out) / "0").rglob("*")
        if p.is_file() and p.name not in metadata_names
    ]
    assert chunk_files == []

    # Still reads back as the fill value.
    group = open_group(out, mode="r")
    level0 = read_zarr_array(group, component="0", io_backend="zarr").squeeze().compute()
    np.testing.assert_array_equal(
        level0.astype(np.uint16), np.zeros((8, 16, 16), dtype=np.uint16)
    )


def test_store_ome_zarr_clamps_levels(tmp_path) -> None:
    data = da.from_array(np.ones((4, 4, 4), dtype=np.uint16), chunks=(4, 4, 4))
    out = str(tmp_path / "tiny.zarr")

    store_ome_zarr(
        data,
        out,
        n_levels=10,  # far more than a 4x4x4 volume supports
        overwrite=True,
        zarr_format=3,
        io_backend="zarr",
        chunks=(1, 1, 4, 4, 4),
    )

    group = open_group(out, mode="r")
    # (4,4,4) -> level 1 (2,2,2); xarray-multiscale stops there (2 levels total).
    assert "0" in group
    assert "1" in group
    assert "2" not in group


def test_store_ome_zarr_skips_existing_when_not_overwrite(tmp_path) -> None:
    data = da.from_array(np.zeros((4, 4, 4), dtype=np.uint16), chunks=(4, 4, 4))
    out = str(tmp_path / "existing.zarr")
    store_ome_zarr(
        data, out, n_levels=1, overwrite=True, zarr_format=3, io_backend="zarr"
    )
    # Second call without overwrite must be a no-op (does not raise).
    store_ome_zarr(
        data, out, n_levels=1, overwrite=False, zarr_format=3, io_backend="zarr"
    )
    assert "0" in open_group(out, mode="r")


@pytest.mark.parametrize(
    "zarr_format, attr_key",
    [(3, "ome"), (2, "multiscales")],
)
def test_parse_ome_zarr_transformations_handles_both_layouts(
    tmp_path, zarr_format, attr_key
) -> None:
    group = open_group(
        str(tmp_path / f"meta_v{zarr_format}.zarr"),
        mode="w",
        zarr_format=zarr_format,
    )
    block = [
        {
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, 1, 1]},
                        {"type": "translation", "translation": [0, 0, 5, 6, 7]},
                    ],
                }
            ]
        }
    ]
    if attr_key == "ome":
        group.attrs["ome"] = {"version": "0.5", "multiscales": block}
    else:
        group.attrs["multiscales"] = block

    transforms = parse_ome_zarr_transformations(group, "0")
    assert transforms["scale"] == [1, 1, 1, 1, 1]
    assert transforms["translation"] == [0, 0, 5, 6, 7]


def test_corrected_reducer_default_is_median_rank() -> None:
    """Sanity check: the corrected/probability default reducer is rank=-2."""
    from exaspim_flatfield_correction.utils.zarr_utils import DEFAULT_REDUCER
    from xarray_multiscale.reducers import windowed_rank

    assert isinstance(DEFAULT_REDUCER, partial)
    assert DEFAULT_REDUCER.func is windowed_rank
    assert DEFAULT_REDUCER.keywords == {"rank": -2}
