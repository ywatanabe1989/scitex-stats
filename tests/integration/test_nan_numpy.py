"""Numpy-path tests for `scitex_stats.descriptive._nan`.

Module sat at 35 % — siblings `test__nan.py` start with
`pytest.importorskip("torch")` and skip when torch isn't installed.
These cover the numpy backend directly.
"""

from __future__ import annotations

import numpy as np

from scitex_stats.descriptive import _nan

_DATA = np.array(
    [
        [1.0, 2.0, np.nan, 4.0],
        [5.0, np.nan, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]
)


def test_nanmax():
    np.testing.assert_array_equal(
        _nan.nanmax(_DATA, axis=-1), np.array([4.0, 8.0, 12.0])
    )


def test_nanmin():
    np.testing.assert_array_equal(
        _nan.nanmin(_DATA, axis=-1), np.array([1.0, 5.0, 9.0])
    )


def test_nansum():
    np.testing.assert_array_equal(
        _nan.nansum(_DATA, axis=-1), np.array([7.0, 20.0, 42.0])
    )


def test_nanmean():
    out = _nan.nanmean(_DATA, axis=-1)
    np.testing.assert_allclose(out, [7 / 3, 20 / 3, 42 / 4])


def test_nanvar_nanstd_consistency():
    var = _nan.nanvar(_DATA, axis=-1)
    std = _nan.nanstd(_DATA, axis=-1)
    np.testing.assert_allclose(std**2, var, atol=1e-9)


def test_nanzscore_centres_to_zero_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 50).astype(float)
    x[3] = np.nan
    # Default `keepdims=True` here matches the signature default.
    z = _nan.nanzscore(x, axis=-1)
    # Load-bearing invariant: the centred output sums to zero
    # (ignoring NaN).
    assert np.isclose(np.nansum(z), 0.0, atol=1e-9)
    assert np.all(np.isfinite(z[~np.isnan(x)]))


def test_nanskewness_and_nankurtosis_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50).astype(float)
    x[5:8] = np.nan
    assert np.isfinite(_nan.nanskewness(x, axis=-1))
    assert np.isfinite(_nan.nankurtosis(x, axis=-1))


def test_nanprod_nancumprod_nancumsum():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert _nan.nanprod(x, axis=-1) == 8.0
    np.testing.assert_array_equal(
        _nan.nancumprod(x, axis=-1), np.array([1.0, 2.0, 2.0, 8.0])
    )
    np.testing.assert_array_equal(
        _nan.nancumsum(x, axis=-1), np.array([1.0, 3.0, 3.0, 7.0])
    )


def test_nanargmin_nanargmax():
    x = np.array([3.0, np.nan, 1.0, 2.0])
    assert _nan.nanargmin(x, axis=-1) == 2
    assert _nan.nanargmax(x, axis=-1) == 0


def test_nanquantile_q25_q50_q75():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 5.0])
    np.testing.assert_allclose(_nan.nanq25(x, axis=-1), 2.0)
    np.testing.assert_allclose(_nan.nanq50(x, axis=-1), 3.0)
    np.testing.assert_allclose(_nan.nanq75(x, axis=-1), 4.0)


def test_nanquantile_with_explicit_q_percent_scale():
    """`q` is on the 0-100 percentile scale (the wrapper divides by
    100 before delegating to np.nanquantile)."""
    x = np.array([0.0, 25.0, 50.0, 75.0, 100.0, np.nan])
    np.testing.assert_allclose(_nan.nanquantile(x, q=50, axis=-1), 50.0)
    np.testing.assert_allclose(_nan.nanquantile(x, q=25, axis=-1), 25.0)


def test_nan_funcs_with_2d_keepdims():
    out = _nan.nanmean(_DATA, axis=-1, keepdims=True)
    assert out.shape == (3, 1)
