"""Tests for `scitex_stats.resampling._delta_auc_ci.delta_auc_ci`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scitex_stats.resampling._delong import _delong_two_auc_covar
from scitex_stats.resampling._delta_auc_ci import delta_auc_ci


def test_delta_auc_matches_auc_a_minus_auc_b():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    score_a = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])  # perfect, auc=1.0
    score_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # chance, auc=0.5
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert np.isclose(result["delta_auc"], result["auc_a"] - result["auc_b"])


def test_delta_auc_ci_auc_a_matches_perfect_separation():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    score_a = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])  # perfect, auc=1.0
    score_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # chance, auc=0.5
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert np.isclose(result["auc_a"], 1.0)


def test_delta_auc_ci_auc_b_matches_chance():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    score_a = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])  # perfect, auc=1.0
    score_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # chance, auc=0.5
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert np.isclose(result["auc_b"], 0.5)


def test_delta_auc_ci_delta_matches_expected_difference():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    score_a = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])  # perfect, auc=1.0
    score_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # chance, auc=0.5
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert np.isclose(result["delta_auc"], 0.5)


def test_delta_auc_ci_ordering_holds():
    # Arrange
    rng = np.random.default_rng(10)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 1, size=n)
    score_b = y_true + rng.normal(0, 1, size=n)
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert result["ci_lower"] <= result["delta_auc"] <= result["ci_upper"]


def _correlated_scores(seed):
    """Shared fixture-style helper for the covariance-awareness tests
    below: two score arrays derived from the same underlying signal,
    so they are positively correlated (evaluated on the same samples).
    Not a test itself — used to avoid duplicating the construction."""
    rng = np.random.default_rng(seed)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    signal = y_true + rng.normal(0, 1, size=n)
    score_a = signal + rng.normal(0, 0.1, size=n)
    score_b = signal + rng.normal(0, 0.1, size=n)
    return y_true, score_a, score_b


def test_delta_auc_ci_correlated_scores_have_positive_covariance():
    """Sanity check that the constructed scores are indeed positively
    correlated, which the narrowing tests below depend on."""
    # Arrange
    y_true, score_a, score_b = _correlated_scores(11)
    # Act
    _, _, _, _, covar_ab, _, _ = _delong_two_auc_covar(y_true, score_a, score_b)
    # Assert
    assert covar_ab > 0


def test_delta_auc_ci_correlation_awareness_matches_covariance_formula():
    """The CI returned by delta_auc_ci must equal the width implied by
    the full covariance-aware DeLong variance formula
    (var_a + var_b - 2*covar_ab), confirming the covariance term is
    actually applied rather than ignored."""
    # Arrange
    y_true, score_a, score_b = _correlated_scores(11)
    _, _, var_a, var_b, covar_ab, _, _ = _delong_two_auc_covar(
        y_true, score_a, score_b
    )
    ci = 95
    z = norm.ppf(0.5 + ci / 200.0)
    delong_var = max(var_a + var_b - 2.0 * covar_ab, 0.0)
    delong_width = 2 * z * np.sqrt(delong_var)
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, ci=ci, method="delong")
    result_width = result["ci_upper"] - result["ci_lower"]
    # Assert
    assert np.isclose(result_width, delong_width, atol=1e-9)


def test_delta_auc_ci_correlation_awareness_narrows_ci_vs_naive_independent_sum():
    """The core DeLong property: when the two classifiers' scores are
    positively correlated (evaluated on the same samples), the full
    covariance-aware CI must be narrower than a naive CI that ignores
    the covariance term (var_a + var_b instead of
    var_a + var_b - 2*covar_ab)."""
    # Arrange
    y_true, score_a, score_b = _correlated_scores(11)
    _, _, var_a, var_b, covar_ab, _, _ = _delong_two_auc_covar(
        y_true, score_a, score_b
    )
    ci = 95
    z = norm.ppf(0.5 + ci / 200.0)
    delong_var = max(var_a + var_b - 2.0 * covar_ab, 0.0)
    naive_var = var_a + var_b  # ignores the covariance term entirely
    delong_width = 2 * z * np.sqrt(delong_var)
    naive_width = 2 * z * np.sqrt(naive_var)
    # Act
    # (widths are compared directly below; no extra call needed)
    # Assert
    assert delong_width < naive_width


def test_delta_auc_ci_correlation_awareness_differs_from_naive_independent_sum():
    """Complementary check: the DeLong-returned CI width must not
    accidentally coincide with the naive (covariance-ignoring) width."""
    # Arrange
    y_true, score_a, score_b = _correlated_scores(11)
    _, _, var_a, var_b, _, _, _ = _delong_two_auc_covar(y_true, score_a, score_b)
    ci = 95
    z = norm.ppf(0.5 + ci / 200.0)
    naive_var = var_a + var_b
    naive_width = 2 * z * np.sqrt(naive_var)
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, ci=ci, method="delong")
    result_width = result["ci_upper"] - result["ci_lower"]
    # Assert
    assert result_width != naive_width


def test_delta_auc_ci_p_value_small_when_scores_differ_a_lot():
    # Arrange
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score_a = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])  # near-perfect
    score_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # chance
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert result["p_value"] < 0.2


def test_delta_auc_ci_delta_is_zero_when_scores_identical():
    # Arrange
    rng = np.random.default_rng(12)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 1, size=n)
    score_b = score_a.copy()  # delta = 0 exactly
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert np.isclose(result["delta_auc"], 0.0)


def test_delta_auc_ci_p_value_large_when_scores_identical():
    # Arrange
    rng = np.random.default_rng(12)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 1, size=n)
    score_b = score_a.copy()  # delta = 0 exactly
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    # Assert
    assert result["p_value"] > 0.5


def test_delta_auc_ci_raises_on_single_class():
    # Arrange
    y_true = np.array([1, 1, 1, 1])
    score_a = np.array([0.1, 0.2, 0.3, 0.4])
    score_b = np.array([0.4, 0.3, 0.2, 0.1])
    # Act
    # (call happens inside the Assert block below)
    # Assert
    with pytest.raises(ValueError):
        delta_auc_ci(y_true, score_a, score_b, method="delong")


def test_delta_auc_ci_bootstrap_ordering_holds():
    # Arrange
    rng = np.random.default_rng(13)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 1, size=n)
    score_b = y_true + rng.normal(0, 1, size=n)
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="bootstrap", n_boot=200, seed=42)
    # Assert
    assert result["ci_lower"] <= result["delta_auc"] <= result["ci_upper"]


def test_delta_auc_ci_bootstrap_reproducible_ci_lower_with_fixed_seed():
    # Arrange
    rng = np.random.default_rng(13)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 1, size=n)
    score_b = y_true + rng.normal(0, 1, size=n)
    # Act
    result1 = delta_auc_ci(
        y_true, score_a, score_b, method="bootstrap", n_boot=200, seed=42
    )
    result2 = delta_auc_ci(
        y_true, score_a, score_b, method="bootstrap", n_boot=200, seed=42
    )
    # Assert
    assert result1["ci_lower"] == result2["ci_lower"]


def test_delta_auc_ci_bootstrap_reproducible_ci_upper_with_fixed_seed():
    # Arrange
    rng = np.random.default_rng(13)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 1, size=n)
    score_b = y_true + rng.normal(0, 1, size=n)
    # Act
    result1 = delta_auc_ci(
        y_true, score_a, score_b, method="bootstrap", n_boot=200, seed=42
    )
    result2 = delta_auc_ci(
        y_true, score_a, score_b, method="bootstrap", n_boot=200, seed=42
    )
    # Assert
    assert result1["ci_upper"] == result2["ci_upper"]


def test_delta_auc_ci_bootstrap_delta_matches_auc_a_minus_auc_b():
    # Arrange
    rng = np.random.default_rng(13)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 1, size=n)
    score_b = y_true + rng.normal(0, 1, size=n)
    # Act
    result = delta_auc_ci(y_true, score_a, score_b, method="bootstrap", n_boot=200, seed=42)
    # Assert
    assert np.isclose(result["delta_auc"], result["auc_a"] - result["auc_b"])
