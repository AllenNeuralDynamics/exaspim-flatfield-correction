"""Tests for the per-channel probability volume write.

The foreground mask is shared across a tile's channels, but the probability
weights are intensity-derived and must be written for *every* tile, including
channels that reuse the cached mask. These tests exercise
``_create_probability_volume`` directly (the unit the fitting stage now calls
per channel) using local ``tmp_path`` zarr stores, so they need neither
tensorstore nor S3.
"""

from __future__ import annotations

import dask.array as da
import numpy as np

from exaspim_flatfield_correction.config import FittingConfig
from exaspim_flatfield_correction.pipeline import _create_probability_volume
from exaspim_flatfield_correction.utils.zarr_utils import store_ome_zarr
from zarr_io.arrays import open_group, read_zarr_array


def _make_tile(path: str, raw: np.ndarray) -> None:
    """Write a minimal input OME-Zarr tile (data + transforms) at ``path``."""
    store_ome_zarr(
        da.from_array(raw, chunks=raw.shape),
        path,
        n_levels=1,
        voxel_size=(2.0, 0.5, 0.5),
        origin=(0, 0, 0, 0, 0),
        overwrite=True,
        zarr_format=3,
        io_backend="zarr",
    )


def _low_res(raw: np.ndarray) -> da.Array:
    return da.from_array(raw.astype(np.float32), chunks=(2, *raw.shape[1:]))


def _write_probability(tmp_path, tile_name, raw, config, prob_root):
    """Build a tile group and write its probability volume via the pipeline."""
    tile_path = str(tmp_path / f"{tile_name}.zarr")
    _make_tile(tile_path, raw)
    z = open_group(tile_path, mode="r")
    return _create_probability_volume(
        low_res=_low_res(raw),
        z=z,
        fitting_res="0",
        tile_name=tile_name,
        out_probability_path=str(prob_root),
        config=config,
        bkg_slice_indices=np.array([0, 1], dtype=np.int64),
        overwrite=True,
        io_backend="zarr",
    )


def test_probability_volume_written_per_channel(tmp_path) -> None:
    """Both channels of a tile get their own probability/<tile> output.

    Regression for the bug where the probability volume was written only for
    the first tile that computed the shared mask, so the second channel had a
    mask but no probability volume.
    """
    config = FittingConfig(enable_gmm_refinement=True)
    prob_root = tmp_path / "probability"

    rng = np.random.default_rng(0)
    raw_a = (rng.random((8, 16, 16)) * 1000).astype(np.float32)
    raw_b = (rng.random((8, 16, 16)) * 1000).astype(np.float32)

    weights_a = _write_probability(
        tmp_path, "tile_000000_ch_488", raw_a, config, prob_root
    )
    weights_b = _write_probability(
        tmp_path, "tile_000000_ch_561", raw_b, config, prob_root
    )

    # Each channel returns a materializable volume matching the input shape.
    assert weights_a is not None and weights_b is not None
    assert weights_a.squeeze().shape == raw_a.shape
    assert weights_b.squeeze().shape == raw_b.shape

    # Both channels are persisted independently under their own tile name.
    for tile_name in ("tile_000000_ch_488", "tile_000000_ch_561"):
        group = open_group(str(prob_root / tile_name), mode="r")
        back = read_zarr_array(
            group, component="0", io_backend="zarr"
        ).squeeze().compute()
        assert back.shape == raw_a.shape
        assert back.dtype == np.float32


def test_probability_volume_skipped_when_refinement_disabled(tmp_path) -> None:
    """No probability volume is produced when GMM refinement is off."""
    config = FittingConfig(enable_gmm_refinement=False)
    prob_root = tmp_path / "probability"
    raw = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)

    weights = _write_probability(
        tmp_path, "tile_000000_ch_488", raw, config, prob_root
    )

    assert weights is None
    assert not (prob_root / "tile_000000_ch_488").exists()
