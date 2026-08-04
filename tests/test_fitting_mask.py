"""Tests for fitting-mask preparation and background-slice exclusion."""

from __future__ import annotations

import dask.array as da
import numpy as np
import zarr

from exaspim_flatfield_correction import pipeline as pipeline_module
from exaspim_flatfield_correction.config import FittingConfig
from exaspim_flatfield_correction.pipeline import (
    MaskArtifacts,
    _filtered_mask_cache_path,
    flatfield_fitting,
)


def test_flatfield_fitting_excludes_background_slices_before_statistics(
    tmp_path,
    monkeypatch,
) -> None:
    """Global/profile statistics and the output mask omit background planes."""
    low_shape = (3, 4, 3)
    full_shape = tuple(2 * size for size in low_shape)
    raw = np.full(low_shape, 20, dtype=np.uint16)
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    input_array = root.create_array(
        "3", shape=low_shape, dtype=raw.dtype, chunks=(2, 4, 3)
    )
    input_array[:] = raw

    full_res = da.full(full_shape, 20, chunks=(2, 4, 3), dtype=np.float32)
    shared_mask = da.ones(low_shape, chunks=(2, 4, 3), dtype=bool)
    background_indices = np.asarray([1, 2], dtype=np.int64)
    expected_low_res_mask = np.ones(low_shape, dtype=bool)
    expected_low_res_mask[background_indices] = False
    expected_full_res_mask = np.repeat(
        np.repeat(
            np.repeat(expected_low_res_mask, 2, axis=0), 2, axis=1
        ),
        2,
        axis=2,
    )
    observed_masks: dict[str, np.ndarray] = {}
    observed_mask_names: dict[str, str] = {}

    monkeypatch.setattr(
        pipeline_module,
        "_create_probability_volume",
        lambda **_: None,
    )

    def fake_weighted_percentile(data, mask, percentile, weights=None):
        observed_masks["global"] = mask.compute()
        observed_mask_names["global"] = mask.name
        return 10.0

    def fake_compute_axis_fits(volume, mask, full_shape, **kwargs):
        observed_masks["profiles"] = mask.compute()
        observed_mask_names["profiles"] = mask.name
        return {}

    def fake_apply_axis_corrections(
        full_res,
        mask_upscaled,
        axis_fits,
        **kwargs,
    ):
        observed_masks["output"] = mask_upscaled.compute()
        return full_res

    monkeypatch.setattr(
        pipeline_module, "weighted_percentile", fake_weighted_percentile
    )
    monkeypatch.setattr(
        pipeline_module, "compute_axis_fits", fake_compute_axis_fits
    )
    monkeypatch.setattr(
        pipeline_module, "apply_axis_corrections", fake_apply_axis_corrections
    )

    corrected, axis_fits, returned_artifacts = flatfield_fitting(
        full_res=full_res,
        z=root,
        is_binned_channel=False,
        mask_dir=str(tmp_path),
        tile_name="tile_ch_561.zarr",
        out_mask_path=str(tmp_path / "mask"),
        out_probability_path=str(tmp_path / "probability"),
        coordinate_transformations={
            "scale": (1.0, 1.0, 1.0, 1.0, 1.0),
            "translation": (0.0, 0.0, 0.0, 0.0, 0.0),
        },
        overwrite=True,
        config=FittingConfig(
            enable_gmm_refinement=False,
            gaussian_sigma=0,
        ),
        bkg=np.zeros(low_shape[1:], dtype=np.float32),
        bkg_slice_indices=background_indices,
        mask_artifacts=MaskArtifacts(mask_low_res=shared_mask),
        results_dir=str(tmp_path),
        io_backend="zarr",
    )

    assert corrected is full_res
    assert axis_fits == {}
    assert returned_artifacts is not None
    np.testing.assert_array_equal(
        returned_artifacts.mask_low_res.compute(),
        np.ones(low_shape, dtype=bool),
    )
    assert set(observed_masks) == {"global", "profiles", "output"}
    np.testing.assert_array_equal(
        observed_masks["global"], expected_low_res_mask
    )
    np.testing.assert_array_equal(
        observed_masks["profiles"], expected_low_res_mask
    )
    np.testing.assert_array_equal(
        observed_masks["output"], expected_full_res_mask
    )
    assert all(
        name.startswith("filtered-mask-cache-")
        for name in observed_mask_names.values()
    )
    assert _filtered_mask_cache_path(
        str(tmp_path), "tile_ch_561.zarr"
    ).is_dir()
