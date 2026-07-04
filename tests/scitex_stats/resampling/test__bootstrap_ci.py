"""Tests for `scitex_stats.resampling._bootstrap_ci.bootstrap_ci`."""

from __future__ import annotations

import numpy as np

from scitex_stats.resampling._bootstrap_ci import bootstrap_ci


def test_bootstrap_ci_estimate_matches_fn_on_original_data():
    # Arrange
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Act
    result = bootstrap_ci(np.mean, data, n_boot=500, seed=0)
    # Assert
    assert np.isclose(result["estimate"], np.mean(data))


def test_bootstrap_ci_ordering_holds_on_gaussian_data():
    # Arrange
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=200)
    # Act
    result = bootstrap_ci(np.mean, data, n_boot=1000, seed=0)
    # Assert
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


def test_bootstrap_ci_reproducible_with_fixed_seed():
    # Arrange
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, size=100)
    # Act
    result1 = bootstrap_ci(np.mean, data, n_boot=300, seed=42)
    result2 = bootstrap_ci(np.mean, data, n_boot=300, seed=42)
    # Assert
    assert result1 == result2


def test_bootstrap_ci_small_n_boot_does_not_crash():
    # Arrange
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Act
    result = bootstrap_ci(np.mean, data, n_boot=10, seed=0)
    # Assert
    assert result["ci_lower"] <= result["ci_upper"]


def test_bootstrap_ci_paired_resampling_preserves_estimate():
    """If arrays were resampled independently (not paired with shared
    indices), `a == b` would not stay True across all iterations, and
    the estimate/CI over that fn would collapse to something other
    than 1.0. With correct paired resampling it must be exactly 1.0
    since `a` and `b` are identical arrays."""
    # Arrange
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = a.copy()
    fn = lambda x, y: np.mean(x == y)
    # Act
    result = bootstrap_ci(fn, a, b, n_boot=300, seed=0)
    # Assert
    assert result["estimate"] == 1.0


def test_bootstrap_ci_paired_resampling_ci_lower_stays_at_one():
    # Arrange
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = a.copy()
    fn = lambda x, y: np.mean(x == y)
    # Act
    result = bootstrap_ci(fn, a, b, n_boot=300, seed=0)
    # Assert
    assert result["ci_lower"] == 1.0


def test_bootstrap_ci_paired_resampling_ci_upper_stays_at_one():
    # Arrange
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = a.copy()
    fn = lambda x, y: np.mean(x == y)
    # Act
    result = bootstrap_ci(fn, a, b, n_boot=300, seed=0)
    # Assert
    assert result["ci_upper"] == 1.0


def test_bootstrap_ci_formatted_string_mentions_estimate():
    # Arrange
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Act
    result = bootstrap_ci(np.mean, data, n_boot=100, seed=0)
    # Assert
    assert "Estimate" in result["formatted"]


def test_bootstrap_ci_formatted_string_mentions_ci():
    # Arrange
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Act
    result = bootstrap_ci(np.mean, data, n_boot=100, seed=0)
    # Assert
    assert "CI" in result["formatted"]
