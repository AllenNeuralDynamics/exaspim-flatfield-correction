"""Tests for fitting helpers (probability-weight computation, etc.)."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

from exaspim_flatfield_correction.fitting import calc_percentile_weight


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
