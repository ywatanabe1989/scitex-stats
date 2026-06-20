"""Numpy-path tests for `scitex_stats.descriptive._real`.

Module sat at 28 % — covers mean / std / var / zscore / skewness /
kurtosis / quantile / q25 / q50 / q75 directly.
"""

from __future__ import annotations

import numpy as np

from scitex_stats.descriptive import _real

_DATA = np.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]
)


def test_mean_array_equal_data_array_real():
    # Arrange
    # Act
    # Assert
    assert np.array_equal(_real.mean(_DATA, axis=-1), np.array([2.5, 6.5, 10.5]), equal_nan=True)


def test_std_var_consistency():
    # Arrange
    s = _real.std(_DATA, axis=-1)
    # Act
    v = _real.var(_DATA, axis=-1)
    # Assert
    assert np.allclose(s ** 2, v, rtol=1e-07, atol=1e-09, equal_nan=True)


def test_zscore_centres_each_row_to_zero_mean():
    # Default keepdims=True is the safe path (matches the signature
    # default). Explicit `keepdims=False` is a known-buggy combo with
    # 2-D input — np.squeeze rejects axes that aren't size 1.
    # Arrange
    # Act
    z = _real.zscore(_DATA, axis=-1)
    # Assert
    assert np.allclose(z.mean(axis=-1), 0.0, rtol=1e-07, atol=1e-09, equal_nan=True)


def test_zscore_keepdims_preserves_shape():
    # Arrange
    # Act
    z = _real.zscore(_DATA, axis=-1, keepdims=True)
    # Assert
    assert z.shape == _DATA.shape


def test_skewness_finite_isfinite():
    # Arrange
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50).astype(float)
    # Act
    out = _real.skewness(x, axis=-1)
    # Assert
    assert np.isfinite(out)


def test_kurtosis_excess_returns_finite():
    # Arrange
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50).astype(float)
    # Act
    out = _real.kurtosis(x, axis=-1)
    # Assert
    assert np.isfinite(out)


def test_quantile_with_explicit_q_percent_scale_allclose_real():
    """`q` is on the 0-100 scale (the wrapper divides by 100)."""
    # Arrange
    # Act
    x = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    # Assert
    assert np.allclose(_real.quantile(x, q=50, axis=-1), 50.0, rtol=1e-07, atol=0, equal_nan=True)

def test_quantile_with_explicit_q_percent_scale_allclose_real_2():
    """`q` is on the 0-100 scale (the wrapper divides by 100)."""
    # Arrange
    # Act
    x = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    # Assert
    assert np.allclose(_real.quantile(x, q=25, axis=-1), 25.0, rtol=1e-07, atol=0, equal_nan=True)


def test_q25_q50_q75_allclose_real():
    # Arrange
    # Act
    x = np.arange(1.0, 6.0)  # [1, 2, 3, 4, 5]
    # Assert
    assert np.allclose(_real.q25(x, axis=-1), 2.0, rtol=1e-07, atol=0, equal_nan=True)

def test_q25_q50_q75_allclose_real_2():
    # Arrange
    # Act
    x = np.arange(1.0, 6.0)  # [1, 2, 3, 4, 5]
    # Assert
    assert np.allclose(_real.q50(x, axis=-1), 3.0, rtol=1e-07, atol=0, equal_nan=True)

def test_q25_q50_q75_allclose_real_3():
    # Arrange
    # Act
    x = np.arange(1.0, 6.0)  # [1, 2, 3, 4, 5]
    # Assert
    assert np.allclose(_real.q75(x, axis=-1), 4.0, rtol=1e-07, atol=0, equal_nan=True)


def test_quantile_2d_keepdims():
    # Arrange
    # Act
    out = _real.quantile(_DATA, q=50, axis=-1, keepdims=True)
    # Assert
    assert out.shape == (3, 1)


def test_quantile_tuple_axis():
    # Arrange
    # Act
    out = _real.quantile(_DATA, q=50, axis=(0, 1))
    # Assert
    assert np.ndim(out) == 0


def test_mean_with_keepdims():
    # Arrange
    # Act
    out = _real.mean(_DATA, axis=-1, keepdims=True)
    # Assert
    assert out.shape == (3, 1)
