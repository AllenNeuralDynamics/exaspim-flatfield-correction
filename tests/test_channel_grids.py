"""Masks for tiles whose channels are sampled at different z steps.

Dataset 853730 has 561 at an 8 um z step and 488 at 2 um, with the same pixel
size, so the two channels' fitting levels differ in z and their stacks end at
different depths. The shared tile mask has to be placed on each channel's grid by
physical position.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr

from exaspim_flatfield_correction import pipeline as pipeline_module
from exaspim_flatfield_correction.config import FittingConfig
from exaspim_flatfield_correction.pipeline import (
    _mask_source_grid,
    _preprocess_mask,
    flatfield_fitting,
)
from exaspim_flatfield_correction.utils.mask_utils import (
    grid_from_transformations,
    resample_mask_to_grid,
    same_physical_extent,
)

# Level-4 and level-3 z grids of tile 0 in 853730 (scale um, translation um).
L4_488 = {"scale": [32.0, 11.968, 11.968], "translation": [24071.5, 0.0, 0.0]}
L3_488 = {"scale": [16.0, 5.984, 5.984], "translation": [24063.5, 0.0, 0.0]}
L4_561 = {"scale": [128.0, 11.968, 11.968], "translation": [24116.5, 0.0, 0.0]}
L3_561 = {"scale": [64.0, 5.984, 5.984], "translation": [24084.5, 0.0, 0.0]}


def test_grid_from_transformations_takes_zyx() -> None:
    grid = grid_from_transformations(
        {"scale": [1, 1, 8, 0.748, 0.748], "translation": [0, 0, 5, 6, 7]}
    )
    assert grid == {"scale": [8.0, 0.748, 0.748], "translation": [5.0, 6.0, 7.0]}
    assert grid_from_transformations({"scale": [2, 3, 4]})["translation"] == [0.0] * 3
    assert grid_from_transformations({}) is None


def test_same_physical_extent_for_a_binned_channel() -> None:
    # 859684: Mask Gen's 561 level 1 against 488 level 3 cover the same box.
    l1_561 = {"scale": [16.0, 11.968, 11.968], "translation": [4865.3, 13806.2, 42120.4]}
    l3_488 = {"scale": [8.0, 5.984, 5.984], "translation": [4864.8, 13805.8, 42120.0]}
    assert same_physical_extent((1280, 665, 887), l1_561, (2560, 1330, 1774), l3_488)


def test_different_extent_for_a_coarser_z_step() -> None:
    # 853730: the 561 stack ends about 1 mm before the 488 stack.
    assert not same_physical_extent((768, 665, 887), L4_488, (368, 1330, 1774), L3_561)
    assert not same_physical_extent((184, 665, 887), L4_561, (1536, 1330, 1774), L3_488)
    assert same_physical_extent((768, 665, 887), L4_488, (1536, 1330, 1774), L3_488)


def test_resample_mask_onto_coarser_z() -> None:
    mask = np.zeros((24, 2, 2), dtype=bool)
    mask[4:12] = True
    grid_fine = {"scale": [32.0, 1.0, 1.0], "translation": [24071.5, 0.0, 0.0]}
    grid_coarse = {"scale": [128.0, 1.0, 1.0], "translation": [24116.5, 0.0, 0.0]}
    out = resample_mask_to_grid(mask, grid_fine, (5, 2, 2), grid_coarse)
    # Coarse slice k sits at fine slice 1.41 + 4k.
    assert out[:, 0, 0].tolist() == [False, True, True, False, False]


def test_resample_mask_is_false_beyond_the_source_stack() -> None:
    coarse = np.ones((5, 2, 2), dtype=bool)
    grid_fine = {"scale": [32.0, 1.0, 1.0], "translation": [24071.5, 0.0, 0.0]}
    grid_coarse = {"scale": [128.0, 1.0, 1.0], "translation": [24116.5, 0.0, 0.0]}
    out = resample_mask_to_grid(coarse, grid_coarse, (24, 2, 2), grid_fine)
    assert out[:, 1, 1].tolist() == [True] * 20 + [False] * 4


def test_preprocess_mask_resizes_by_shape_when_extents_match(tmp_path, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("same extents must keep the resize by shape")

    monkeypatch.setattr(pipeline_module, "resample_mask_to_grid", fail)
    mask = np.zeros((4, 2, 2), dtype=bool)
    mask[1:3] = True
    out = _preprocess_mask(
        mask,
        (8, 4, 4),
        str(tmp_path),
        "tile",
        chunks=(8, 4, 4),
        source_grid={"scale": [2.0, 2.0, 2.0], "translation": [0.5, 0.5, 0.5]},
        target_grid={"scale": [1.0, 1.0, 1.0], "translation": [0.0, 0.0, 0.0]},
    ).compute()
    assert out[:, 0, 0].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


# A miniature of tile 0 in 853730: 488 at a 2 um z step, 561 at 8 um, same pixel.
# The 561 stack is 128 um shallower than the 488 stack.
GRIDS = {
    "488": {
        "3": ((24, 4, 4), [16.0, 6.0, 6.0], [7.0, 2.625, 2.625]),
        "4": ((12, 2, 2), [32.0, 12.0, 12.0], [15.0, 5.625, 5.625]),
    },
    "561": {
        "3": ((4, 4, 4), [64.0, 6.0, 6.0], [28.0, 2.625, 2.625]),
        "4": ((2, 2, 2), [128.0, 12.0, 12.0], [60.0, 5.625, 5.625]),
    },
}
# Tissue between 63 and 255 um: 488 level-4 slices 2..7.
EXPECTED_Z = {
    "488": [False] * 4 + [True] * 12 + [False] * 8,
    "561": [False, True, True, True],
}


def _write_tile(root, channel):
    path = root / f"tile_000000_ch_{channel}.ome.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    datasets = []
    for level, (shape, scale, translation) in GRIDS[channel].items():
        array = group.create_array(level, shape=shape, dtype=np.uint16, chunks=shape)
        array[:] = 20
        datasets.append(
            {
                "path": level,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            }
        )
    group.attrs["ome"] = {"version": "0.5", "multiscales": [{"datasets": datasets}]}
    return str(path)


def test_mask_source_grid_finds_the_mask_gen_level(tmp_path) -> None:
    paths = [_write_tile(tmp_path, "561"), _write_tile(tmp_path, "488")]
    mask_path = str(tmp_path / "SPIM.ome.zarr_tile_000000_ch_488.ome.zarr_pred.tif")
    grid = _mask_source_grid(mask_path, (12, 2, 2), paths)
    assert grid == {"scale": [32.0, 12.0, 12.0], "translation": [15.0, 5.625, 5.625]}
    assert _mask_source_grid(mask_path, (7, 7, 7), paths) is None
    assert _mask_source_grid(mask_path, (12, 2, 2), None) is None


@pytest.mark.parametrize("order", [("488", "561"), ("561", "488")])
def test_each_channel_gets_the_mask_at_the_same_depth(tmp_path, monkeypatch, order) -> None:
    paths = {channel: _write_tile(tmp_path, channel) for channel in ("488", "561")}
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    mask_gen = np.zeros((12, 2, 2), dtype=np.uint8)
    mask_gen[2:8] = 1
    tifffile.imwrite(
        mask_dir / "SPIM.ome.zarr_tile_000000_ch_488.ome.zarr_pred.tif", mask_gen
    )

    observed: dict[str, np.ndarray] = {}
    current: dict[str, str] = {}
    monkeypatch.setattr(pipeline_module, "_create_probability_volume", lambda **_: None)

    def fake_weighted_percentile(data, mask, percentile, weights=None):
        assert mask.shape == data.shape
        observed[current["channel"]] = mask.compute()
        return 10.0

    def fake_compute_axis_fits(volume, mask, full_shape, **kwargs):
        assert mask.shape == volume.shape
        return {}

    monkeypatch.setattr(pipeline_module, "weighted_percentile", fake_weighted_percentile)
    monkeypatch.setattr(pipeline_module, "compute_axis_fits", fake_compute_axis_fits)
    monkeypatch.setattr(
        pipeline_module,
        "apply_axis_corrections",
        lambda full_res, mask_upscaled, axis_fits, **kwargs: full_res,
    )

    artifacts = None
    for channel in order:
        current["channel"] = channel
        low_shape = GRIDS[channel]["3"][0]
        full_res = da.full(
            tuple(2 * n for n in low_shape), 20, chunks=low_shape, dtype=np.float32
        )
        _, _, artifacts = flatfield_fitting(
            full_res=full_res,
            z=zarr.open_group(paths[channel], mode="r"),
            is_binned_channel=False,
            mask_dir=str(mask_dir),
            tile_name=f"tile_000000_ch_{channel}.ome.zarr",
            out_mask_path=str(tmp_path / "mask"),
            out_probability_path=str(tmp_path / "probability"),
            coordinate_transformations={
                "scale": (1.0, 1.0, 1.0),
                "translation": (0.0, 0.0, 0.0),
            },
            overwrite=True,
            config=FittingConfig(enable_gmm_refinement=False, gaussian_sigma=0),
            bkg=np.zeros(low_shape[1:], dtype=np.float32),
            bkg_slice_indices=np.asarray([0], dtype=np.int64),
            mask_artifacts=artifacts,
            results_dir=str(tmp_path / "results"),
            io_backend="zarr",
            tile_paths=list(paths.values()),
        )
        assert artifacts is not None

    for channel in ("488", "561"):
        mask = observed[channel]
        assert mask.shape == GRIDS[channel]["3"][0]
        assert mask[:, 1, 2].tolist() == EXPECTED_Z[channel], channel
        assert (mask == mask[:, :1, :1]).all()


def test_reusing_a_mask_of_another_shape_needs_grids(tmp_path, monkeypatch) -> None:
    path = _write_tile(tmp_path, "561")
    monkeypatch.setattr(pipeline_module, "_create_probability_volume", lambda **_: None)
    with pytest.raises(ValueError, match="OME scale metadata"):
        flatfield_fitting(
            full_res=da.full((8, 8, 8), 20, chunks=(4, 4, 4), dtype=np.float32),
            z=zarr.open_group(path, mode="r"),
            is_binned_channel=False,
            mask_dir=str(tmp_path),
            tile_name="tile_000000_ch_561.ome.zarr",
            out_mask_path=str(tmp_path / "mask"),
            out_probability_path=str(tmp_path / "probability"),
            coordinate_transformations={"scale": (1.0,) * 3, "translation": (0.0,) * 3},
            overwrite=True,
            config=FittingConfig(enable_gmm_refinement=False, gaussian_sigma=0),
            bkg=np.zeros((4, 4), dtype=np.float32),
            bkg_slice_indices=np.asarray([0], dtype=np.int64),
            mask_artifacts=pipeline_module.MaskArtifacts(
                mask_low_res=da.ones((24, 4, 4), chunks=(24, 4, 4), dtype=bool)
            ),
            results_dir=str(tmp_path / "results"),
            io_backend="zarr",
        )
