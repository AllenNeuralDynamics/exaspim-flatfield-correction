"""Tests for the local disk caches used by the fitting pipeline stage."""

from __future__ import annotations

import dask.array as da
import numpy as np

from exaspim_flatfield_correction.pipeline import (
    _cache_dask_array,
    _fitting_cache_dir,
    _low_res_cache_path,
)


def test_cache_dask_array_roundtrip_float32(tmp_path) -> None:
    rng = np.random.default_rng(0)
    data = (rng.random((8, 10, 12)) * 1000).astype(np.float32)
    arr = da.from_array(data, chunks=(4, 5, 12))
    cache_path = tmp_path / "low_res.zarr"

    cached = _cache_dask_array(arr, cache_path, name="lowres-cache-test")

    assert cache_path.exists()
    assert cached.dtype == np.float32
    assert cached.chunks == arr.chunks
    np.testing.assert_array_equal(cached.compute(), data)


def test_cache_dask_array_roundtrip_bool_and_overwrite(tmp_path) -> None:
    cache_path = tmp_path / "mask.zarr"
    first = da.from_array(np.zeros((4, 6), dtype=bool), chunks=(2, 6))
    _cache_dask_array(first, cache_path, name="mask-cache-a")

    mask = np.zeros((4, 6), dtype=bool)
    mask[:, :3] = True
    cached = _cache_dask_array(
        da.from_array(mask, chunks=(2, 6)), cache_path, name="mask-cache-b"
    )

    assert cached.dtype == bool
    np.testing.assert_array_equal(cached.compute(), mask)


def test_cache_deletion_does_not_affect_independent_graphs(tmp_path) -> None:
    """Downstream arrays that never reference the cache survive its removal."""
    import shutil

    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    cached = _cache_dask_array(
        da.from_array(data, chunks=(1, 3, 4)),
        tmp_path / "tmp.zarr",
        name="cache-tmp",
    )
    derived_np = (cached * 2).compute()  # materialized while cache exists
    independent = da.from_array(data, chunks=(1, 3, 4)) + 1

    shutil.rmtree(tmp_path / "tmp.zarr")

    np.testing.assert_array_equal(independent.compute(), data + 1)
    np.testing.assert_array_equal(derived_np, data * 2)


def test_cache_paths_are_safe_and_scoped(tmp_path) -> None:
    results_dir = str(tmp_path)
    path = _low_res_cache_path(results_dir, "tile_x_0000_y_0001_z_0000_ch_488")

    assert _fitting_cache_dir(results_dir) in path.parents
    assert str(path).startswith(str(tmp_path / "dask-temp" / "fitting-cache"))
    # tile names with unsafe characters are sanitized
    weird = _low_res_cache_path(results_dir, "tile with/slash")
    assert "/" not in weird.name.replace(".zarr", "")
