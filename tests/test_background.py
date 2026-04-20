from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

import dask.array as da
import zarr

from exaspim_flatfield_correction.background import (
    estimate_bkg_from_mapped_slices,
    map_slice_indices_to_target,
)
from exaspim_flatfield_correction.pipeline import (
    background_subtraction,
    save_method_outputs,
)


def _transform(z_scale: float, z_translation: float = 0.0) -> dict:
    return {
        "scale": [1.0, 1.0, z_scale, 1.0, 1.0],
        "translation": [0.0, 0.0, z_translation, 0.0, 0.0],
    }


def _write_multiscale_metadata(root: zarr.Group) -> None:
    root.attrs["multiscales"] = [
        {
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": _transform(1)["scale"]},
                        {
                            "type": "translation",
                            "translation": _transform(1)["translation"],
                        },
                    ],
                },
                {
                    "path": "3",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": _transform(2)["scale"]},
                        {
                            "type": "translation",
                            "translation": _transform(2)["translation"],
                        },
                    ],
                },
            ]
        }
    ]


def test_map_slice_indices_identity_with_metadata() -> None:
    mapped = map_slice_indices_to_target(
        np.asarray([0, 2]),
        (4, 5, 6),
        (4, 5, 6),
        source_transformations=_transform(1),
        target_transformations=_transform(1),
    )

    np.testing.assert_array_equal(mapped, np.asarray([0, 2]))


def test_map_slice_indices_level3_to_level0_full_slab() -> None:
    mapped = map_slice_indices_to_target(
        np.asarray([2]),
        (4, 5, 6),
        (32, 40, 48),
        source_transformations=_transform(8),
        target_transformations=_transform(1),
    )

    np.testing.assert_array_equal(mapped, np.arange(16, 24))


def test_map_slice_indices_honors_translation_metadata() -> None:
    mapped = map_slice_indices_to_target(
        np.asarray([2]),
        (4, 5, 6),
        (32, 40, 48),
        source_transformations=_transform(8, z_translation=4),
        target_transformations=_transform(1),
    )

    np.testing.assert_array_equal(mapped, np.arange(20, 28))


def test_estimate_bkg_from_mapped_slices_uses_target_planes() -> None:
    data = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    target = da.from_array(data, chunks=(1, 1, 3))

    bkg, target_indices = estimate_bkg_from_mapped_slices(
        target,
        np.asarray([1]),
        (2, 2, 3),
        source_transformations=_transform(2),
        target_transformations=_transform(1),
        max_spatial_chunk=1,
    )

    np.testing.assert_array_equal(target_indices, np.asarray([2, 3]))
    np.testing.assert_array_equal(bkg, np.median(data[[2, 3]], axis=0))


def test_background_subtraction_builds_background_from_target_planes() -> None:
    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store)

    full_res_data = np.asarray(
        [
            np.full((2, 2), 10, dtype=np.float32),
            np.full((2, 2), 20, dtype=np.float32),
            np.full((2, 2), 30, dtype=np.float32),
            np.full((2, 2), 50, dtype=np.float32),
        ]
    )
    low_res_data = np.asarray(
        [
            [[0, 100], [0, 100]],
            [[1000, 1000], [1000, 1000]],
        ],
        dtype=np.float32,
    )

    root.create_dataset("0", data=full_res_data, chunks=(1, 2, 2))
    root.create_dataset("3", data=low_res_data, chunks=(1, 2, 2))
    _write_multiscale_metadata(root)

    full_res = da.from_zarr(root["0"]).astype(np.float32)
    corrected, bkg, bkg_slice_indices = background_subtraction(
        "synthetic_tile.zarr",
        full_res,
        root,
        is_binned_channel=False,
        background_smoothing_sigma=0,
        target_resolution="0",
    )

    expected_bkg = np.full((2, 2), 40, dtype=np.float32)
    np.testing.assert_array_equal(bkg, expected_bkg)
    np.testing.assert_array_equal(bkg_slice_indices, np.asarray([1]))
    assert np.max(bkg_slice_indices) < root["3"].shape[0]
    np.testing.assert_array_equal(
        corrected.compute(),
        np.clip(full_res_data - expected_bkg, 0, None),
    )


def test_save_method_outputs_writes_background_and_indices(tmp_path) -> None:
    bkg = np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    indices = np.asarray([1, 3, 5], dtype=np.int64)

    save_method_outputs(
        "fitting",
        "tile_000001",
        str(tmp_path),
        True,
        bkg=bkg,
        bkg_slice_indices=indices,
    )

    bkg_path = tmp_path / "tile_000001_bkg.tif"
    indices_path = tmp_path / "tile_000001_bkg_slice_indices.json"

    assert bkg_path.is_file()
    assert indices_path.is_file()
    np.testing.assert_array_equal(tifffile.imread(bkg_path), bkg)
    assert json.loads(indices_path.read_text()) == [1, 3, 5]
