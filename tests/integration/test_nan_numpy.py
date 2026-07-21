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


def test_nanmax_array_equal_data_array_nan():
    # Arrange
    # Act
    # Assert
    assert np.array_equal(_nan.nanmax(_DATA, axis=-1), np.array([4.0, 8.0, 12.0]), equal_nan=True)


def test_nanmin_array_equal_data_array_nan():
    # Arrange
    # Act
    # Assert
    assert np.array_equal(_nan.nanmin(_DATA, axis=-1), np.array([1.0, 5.0, 9.0]), equal_nan=True)


def test_nansum_array_equal_data_array_nan():
    # Arrange
    # Act
    # Assert
    assert np.array_equal(_nan.nansum(_DATA, axis=-1), np.array([7.0, 20.0, 42.0]), equal_nan=True)


def test_nanmean_allclose_case():
    # Arrange
    # Act
    out = _nan.nanmean(_DATA, axis=-1)
    # Assert
    assert np.allclose(out, [7 / 3, 20 / 3, 42 / 4], rtol=1e-07, atol=0, equal_nan=True)


def test_nanvar_nanstd_consistency():
    # Arrange
    var = _nan.nanvar(_DATA, axis=-1)
    # Act
    std = _nan.nanstd(_DATA, axis=-1)
    # Assert
    assert np.allclose(std ** 2, var, rtol=1e-07, atol=1e-09, equal_nan=True)


def test_nanzscore_centres_to_zero_mean_isclose_nansum():
    # Arrange
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 50).astype(float)
    x[3] = np.nan
    # Act
    z = _nan.nanzscore(x, axis=-1)
    # Assert
    assert np.isclose(np.nansum(z), 0.0, atol=1e-9)

def test_nanzscore_centres_to_zero_mean_all_isfinite_isnan():
    # Arrange
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 50).astype(float)
    x[3] = np.nan
    # Act
    z = _nan.nanzscore(x, axis=-1)
    # Assert
    assert np.all(np.isfinite(z[~np.isnan(x)]))


def test_nanskewness_and_nankurtosis_finite_isfinite_nan():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    x = rng.normal(0, 1, 50).astype(float)
    x[5:8] = np.nan
    # Assert
    assert np.isfinite(_nan.nanskewness(x, axis=-1))

def test_nanskewness_and_nankurtosis_finite_isfinite_nan_2():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    x = rng.normal(0, 1, 50).astype(float)
    x[5:8] = np.nan
    # Assert
    assert np.isfinite(_nan.nankurtosis(x, axis=-1))


def test_nanprod_nancumprod_nancumsum_nan():
    # Arrange
    # Act
    x = np.array([1.0, 2.0, np.nan, 4.0])
    # Assert
    assert _nan.nanprod(x, axis=-1) == 8.0

def test_nanprod_nancumprod_nancumsum_array_equal_array_nan():
    # Arrange
    # Act
    x = np.array([1.0, 2.0, np.nan, 4.0])
    # Assert
    assert np.array_equal(_nan.nancumprod(x, axis=-1), np.array([1.0, 2.0, 2.0, 8.0]), equal_nan=True)

def test_nanprod_nancumprod_nancumsum_array_equal_array_nan_2():
    # Arrange
    # Act
    x = np.array([1.0, 2.0, np.nan, 4.0])
    # Assert
    assert np.array_equal(_nan.nancumsum(x, axis=-1), np.array([1.0, 3.0, 3.0, 7.0]), equal_nan=True)


def test_nanargmin_nanargmax_nan():
    # Arrange
    # Act
    x = np.array([3.0, np.nan, 1.0, 2.0])
    # Assert
    assert _nan.nanargmin(x, axis=-1) == 2

def test_nanargmin_nanargmax_nan_2():
    # Arrange
    # Act
    x = np.array([3.0, np.nan, 1.0, 2.0])
    # Assert
    assert _nan.nanargmax(x, axis=-1) == 0


def test_nanquantile_q25_q50_q75_allclose_nanq25_nan():
    # Arrange
    # Act
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 5.0])
    # Assert
    assert np.allclose(_nan.nanq25(x, axis=-1), 2.0, rtol=1e-07, atol=0, equal_nan=True)

def test_nanquantile_q25_q50_q75_allclose_nanq50_nan():
    # Arrange
    # Act
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 5.0])
    # Assert
    assert np.allclose(_nan.nanq50(x, axis=-1), 3.0, rtol=1e-07, atol=0, equal_nan=True)

def test_nanquantile_q25_q50_q75_allclose_nanq75_nan():
    # Arrange
    # Act
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 5.0])
    # Assert
    assert np.allclose(_nan.nanq75(x, axis=-1), 4.0, rtol=1e-07, atol=0, equal_nan=True)


def test_nanquantile_with_explicit_q_percent_scale_allclose_nan():
    """`q` is on the 0-100 percentile scale (the wrapper divides by
    100 before delegating to np.nanquantile)."""
    # Arrange
    # Act
    x = np.array([0.0, 25.0, 50.0, 75.0, 100.0, np.nan])
    # Assert
    assert np.allclose(_nan.nanquantile(x, q=50, axis=-1), 50.0, rtol=1e-07, atol=0, equal_nan=True)

def test_nanquantile_with_explicit_q_percent_scale_allclose_nan_2():
    """`q` is on the 0-100 percentile scale (the wrapper divides by
    100 before delegating to np.nanquantile)."""
    # Arrange
    # Act
    x = np.array([0.0, 25.0, 50.0, 75.0, 100.0, np.nan])
    # Assert
    assert np.allclose(_nan.nanquantile(x, q=25, axis=-1), 25.0, rtol=1e-07, atol=0, equal_nan=True)


def test_nan_funcs_with_2d_keepdims():
    # Arrange
    # Act
    out = _nan.nanmean(_DATA, axis=-1, keepdims=True)
    # Assert
    assert out.shape == (3, 1)
