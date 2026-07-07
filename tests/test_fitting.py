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


def test_compute_axis_fits_returns_float32_curves() -> None:
    """Curves must be float32 so they never promote the volume to float64."""
    volume, mask = _synthetic_volume_and_mask()

    fits = compute_axis_fits(volume, mask, full_shape=(16, 20, 24))

    assert set(fits) == {"x", "y", "z"}
    for fit in fits.values():
        assert fit.dtype == np.float32


def test_apply_axis_corrections_stays_float32_with_float64_fits() -> None:
    """Even float64 curves must not promote the corrected volume."""
    shape = (8, 10, 12)
    full_res = da.ones(shape, dtype=np.float32, chunks=shape)
    mask = da.from_array(np.ones(shape, dtype=bool), chunks=shape)
    fits = {
        "x": np.linspace(0.8, 1.2, shape[2]),  # float64
        "y": np.linspace(0.9, 1.1, shape[1]),  # float64
    }

    corrected = apply_axis_corrections(
        full_res, mask, fits, global_factor=500.0, global_med=400.0
    )

    assert corrected.dtype == np.float32
    assert corrected.compute().dtype == np.float32


def test_apply_axis_corrections_values_and_mask_respected() -> None:
    """Masked voxels get full * ratio / (fx*fy*fz); unmasked stay untouched."""
    rng = np.random.default_rng(3)
    shape = (6, 8, 10)
    data = (rng.random(shape) * 1000).astype(np.float32)
    mask_np = np.zeros(shape, dtype=bool)
    mask_np[:, :4, :] = True
    full_res = da.from_array(data, chunks=shape)
    mask = da.from_array(mask_np, chunks=shape)
    fit_x = np.linspace(0.8, 1.2, shape[2]).astype(np.float32)
    fit_y = np.linspace(0.9, 1.1, shape[1]).astype(np.float32)
    fit_z = np.linspace(0.7, 1.3, shape[0]).astype(np.float32)
    global_factor, global_med = 500.0, 400.0

    corrected = apply_axis_corrections(
        full_res,
        mask,
        {"x": fit_x, "y": fit_y, "z": fit_z},
        global_factor=global_factor,
        global_med=global_med,
    ).compute()

    ratio = global_factor / global_med
    expected = (
        data
        * ratio
        / fit_z.reshape(-1, 1, 1)
        / fit_y.reshape(1, -1, 1)
        / fit_x.reshape(1, 1, -1)
    )
    expected = np.where(mask_np, expected, data)
    expected = np.clip(expected, 0, 2**16 - 1)
    np.testing.assert_allclose(corrected, expected, rtol=1e-5)
    np.testing.assert_array_equal(corrected[~mask_np], data[~mask_np])


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
