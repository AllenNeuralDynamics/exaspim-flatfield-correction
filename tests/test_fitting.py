"""Tests for fitting helpers (probability-weight computation, etc.)."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

from exaspim_flatfield_correction.fitting import (
    apply_axis_corrections,
    calc_percentile_weight,
    compute_axis_fits,
)


@pytest.mark.parametrize("smooth_sigma", [0.0, 3.0])
def test_calc_percentile_weight_is_float32(smooth_sigma) -> None:
    """The returned weights must actually compute as float32, not just declare it.

    Regression test: under NEP 50 (NumPy 2.x) the ``np.float64`` scalar ``B`` in
    the Richards block promoted the float32 block to float64, so the dask array
    *declared* float32 while *computing* float64. The trailing ``.astype`` was a
    silent no-op (meta already float32), and the float64 chunks failed the
    ``safe`` cast into a float32 zarr store. Guard both the declared and the
    actually-computed dtype.
    """
    rng = np.random.default_rng(0)
    img = da.from_array(
        (rng.random((32, 128, 128)) * 1000).astype(np.float32),
        chunks=(16, 128, 128),
    )
    bg = (rng.random((8, 128, 128)) * 1000).astype(np.float32)

    weights = calc_percentile_weight(
        img,
        bg,
        low_percentile=50.0,
        high_percentile=99.9,
        eps=0.001,
        smooth_sigma=smooth_sigma,
    )
    computed = weights.compute()

    assert weights.dtype == np.float32
    assert computed.dtype == np.float32
    assert computed.min() >= 0.0
    assert computed.max() <= 1.0


def test_calc_percentile_weight_degenerate_is_float32() -> None:
    """The degenerate (zero-width) percentile branch also stays float32."""
    img = da.from_array(np.full((32, 128, 128), 5.0, np.float32), chunks=(16, 128, 128))
    bg = np.full((8, 128, 128), 5.0, np.float32)

    weights = calc_percentile_weight(img, bg, smooth_sigma=0.0)

    assert weights.dtype == np.float32
    assert weights.compute().dtype == np.float32


def _synthetic_volume_and_mask() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    volume = (rng.random((8, 10, 12)) * 1000 + 100).astype(np.float32)
    mask = np.ones_like(volume, dtype=bool)
    return volume, mask


def test_compute_axis_fits_skips_noop_limits() -> None:
    """An axis whose limits are (1.0, 1.0) is omitted from the fits."""
    volume, mask = _synthetic_volume_and_mask()

    fits = compute_axis_fits(
        volume,
        mask,
        full_shape=(16, 20, 24),
        limits_x=(0.5, 5.0),
        limits_y=(1.0, 1.0),
        limits_z=(0.5, 1.0),
    )

    assert set(fits) == {"x", "z"}
    assert fits["x"].size == 24
    assert fits["z"].size == 16


def test_compute_axis_fits_all_noop_returns_empty() -> None:
    """All axes disabled via (1.0, 1.0) limits yields an empty mapping."""
    volume, mask = _synthetic_volume_and_mask()

    fits = compute_axis_fits(
        volume,
        mask,
        full_shape=(16, 20, 24),
        limits_x=(1.0, 1.0),
        limits_y=(1.0, 1.0),
        limits_z=(1.0, 1.0),
    )

    assert fits == {}


def test_compute_axis_fits_keeps_constant_non_unit_limits() -> None:
    """Limits like (0.5, 0.5) are a constant correction, not a no-op."""
    volume, mask = _synthetic_volume_and_mask()

    fits = compute_axis_fits(
        volume,
        mask,
        full_shape=(16, 20, 24),
        limits_x=(0.5, 0.5),
        limits_y=(1.0, 1.0),
        limits_z=None,
    )

    assert set(fits) == {"x", "z"}
    np.testing.assert_array_equal(fits["x"], np.full(24, 0.5, np.float32))


def test_apply_axis_corrections_omitted_axis_matches_all_ones_fit() -> None:
    """Omitting an axis from axis_fits is equivalent to an all-ones fit."""
    rng = np.random.default_rng(7)
    shape = (8, 10, 12)
    full_res = da.from_array(
        (rng.random(shape) * 1000).astype(np.float32), chunks=shape
    )
    mask = da.from_array(np.ones(shape, dtype=bool), chunks=shape)
    fit_x = np.linspace(0.8, 1.2, shape[2]).astype(np.float32)

    kwargs = dict(global_factor=500.0, global_med=400.0)
    partial = apply_axis_corrections(full_res, mask, {"x": fit_x}, **kwargs)
    explicit = apply_axis_corrections(
        full_res,
        mask,
        {
            "x": fit_x,
            "y": np.ones(shape[1], np.float32),
            "z": np.ones(shape[0], np.float32),
        },
        **kwargs,
    )

    np.testing.assert_array_equal(partial.compute(), explicit.compute())
