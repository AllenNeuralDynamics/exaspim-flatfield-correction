"""Tests for the local disk caches used by the fitting pipeline stage."""

from __future__ import annotations

import dask.array as da
import numpy as np

from exaspim_flatfield_correction.pipeline import (
    _cache_dask_array,
    _exclude_background_slices_from_mask,
    _filtered_mask_cache_path,
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
    filtered_mask = _filtered_mask_cache_path(
        results_dir, "tile with/slash"
    )
    assert _fitting_cache_dir(results_dir) in filtered_mask.parents
    assert "/" not in filtered_mask.name.replace(".zarr", "")


def test_exclude_background_slices_from_mask() -> None:
    mask_np = np.ones((6, 4, 5), dtype=bool)
    mask_np[4, 1:, 2:] = False

    filtered = _exclude_background_slices_from_mask(
        mask_np,
        np.asarray([1, 3, 3], dtype=np.int64),
    )
    expected = mask_np.copy()
    expected[[1, 3]] = False

    assert filtered.dtype == bool
    assert not np.shares_memory(filtered, mask_np)
    np.testing.assert_array_equal(filtered, expected)


def test_background_slice_exclusion_does_not_mutate_shared_mask() -> None:
    mask_np = np.ones((4, 3, 2), dtype=bool)

    first_channel = _exclude_background_slices_from_mask(
        mask_np, np.asarray([0])
    )
    second_channel = _exclude_background_slices_from_mask(
        mask_np, np.asarray([2])
    )

    assert mask_np.all()
    assert not first_channel[0].any()
    assert first_channel[2].all()
    assert second_channel[0].all()
    assert not second_channel[2].any()


def test_exclude_no_background_slices_returns_independent_mask() -> None:
    mask = np.ones((2, 3, 4), dtype=np.uint8)

    filtered = _exclude_background_slices_from_mask(
        mask, np.asarray([], dtype=np.int64)
    )

    assert filtered.dtype == bool
    assert not np.shares_memory(filtered, mask)
    assert filtered.all()


def test_exclude_background_slices_rejects_out_of_range_indices() -> None:
    mask = np.ones((4, 3, 2), dtype=bool)

    with np.testing.assert_raises_regex(
        ValueError, "outside the fitting mask Z range"
    ):
        _exclude_background_slices_from_mask(mask, np.asarray([-1, 4]))


def test_exclude_background_slices_requires_3d_mask() -> None:
    with np.testing.assert_raises_regex(
        ValueError, "must be three-dimensional"
    ):
        _exclude_background_slices_from_mask(
            np.ones((3, 4), dtype=bool), np.asarray([0])
        )
