"""Tests for `scitex_stats.resampling._auc_ci.auc_ci`."""

from __future__ import annotations

import numpy as np
import pytest

from scitex_stats.resampling._auc_ci import auc_ci


def test_auc_ci_perfect_separation_delong_gives_auc_one():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    # Act
    result = auc_ci(y_true, y_score, method="delong")
    # Assert
    assert np.isclose(result["auc"], 1.0)


def test_auc_ci_perfect_separation_delong_ci_ordering_holds():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    # Act
    result = auc_ci(y_true, y_score, method="delong")
    # Assert
    assert result["ci_lower"] <= result["auc"] <= result["ci_upper"]


def test_auc_ci_perfect_separation_delong_reports_method():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    # Act
    result = auc_ci(y_true, y_score, method="delong")
    # Assert
    assert result["method"] == "delong"


def test_auc_ci_perfect_separation_delong_counts_samples_correctly():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    # Act
    result = auc_ci(y_true, y_score, method="delong")
    # Assert
    assert (result["n_pos"], result["n_neg"]) == (3, 3)


def test_auc_ci_perfect_separation_delong_formatted_string_mentions_auc():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    # Act
    result = auc_ci(y_true, y_score, method="delong")
    # Assert
    assert "AUC" in result["formatted"]


def test_auc_ci_perfect_separation_bootstrap_gives_auc_one():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    # Act
    result = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=42)
    # Assert
    assert np.isclose(result["auc"], 1.0)


def test_auc_ci_perfect_separation_bootstrap_ci_ordering_holds():
    # Arrange
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    # Act
    result = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=42)
    # Assert
    assert result["ci_lower"] <= result["auc"] <= result["ci_upper"]


def test_auc_ci_complete_overlap_gives_half_delong():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    # Act
    result = auc_ci(y_true, y_score, method="delong")
    # Assert
    assert np.isclose(result["auc"], 0.5)


def test_auc_ci_ordering_holds_on_random_data_delong():
    # Arrange
    rng = np.random.default_rng(1)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_score = y_true + rng.normal(0, 1, size=n)
    # Act
    result = auc_ci(y_true, y_score, method="delong")
    # Assert
    assert result["ci_lower"] <= result["auc"] <= result["ci_upper"]


def test_auc_ci_ordering_holds_on_random_data_bootstrap():
    # Arrange
    rng = np.random.default_rng(1)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    y_score = y_true + rng.normal(0, 1, size=n)
    # Act
    result = auc_ci(y_true, y_score, method="bootstrap", n_boot=300, seed=7)
    # Assert
    assert result["ci_lower"] <= result["auc"] <= result["ci_upper"]


def test_auc_ci_bootstrap_reproducible_ci_lower_with_fixed_seed():
    # Arrange
    rng = np.random.default_rng(2)
    n = 60
    y_true = rng.integers(0, 2, size=n)
    y_score = y_true + rng.normal(0, 1, size=n)
    # Act
    result1 = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=42)
    result2 = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=42)
    # Assert
    assert result1["ci_lower"] == result2["ci_lower"]


def test_auc_ci_bootstrap_reproducible_ci_upper_with_fixed_seed():
    # Arrange
    rng = np.random.default_rng(2)
    n = 60
    y_true = rng.integers(0, 2, size=n)
    y_score = y_true + rng.normal(0, 1, size=n)
    # Act
    result1 = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=42)
    result2 = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=42)
    # Assert
    assert result1["ci_upper"] == result2["ci_upper"]


def test_auc_ci_bootstrap_can_differ_with_different_seed():
    # Arrange
    rng = np.random.default_rng(3)
    n = 60
    y_true = rng.integers(0, 2, size=n)
    y_score = y_true + rng.normal(0, 1.5, size=n)
    # Act
    result1 = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=1)
    result2 = auc_ci(y_true, y_score, method="bootstrap", n_boot=200, seed=2)
    # Assert
    assert (result1["ci_lower"], result1["ci_upper"]) != (
        result2["ci_lower"],
        result2["ci_upper"],
    )


def test_auc_ci_raises_on_single_class_delong():
    # Arrange
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    # Act
    # (call happens inside the Assert block below)
    # Assert
    with pytest.raises(ValueError):
        auc_ci(y_true, y_score, method="delong")


def test_auc_ci_raises_on_single_class_bootstrap():
    # Arrange
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    # Act
    # (call happens inside the Assert block below)
    # Assert
    with pytest.raises(ValueError):
        auc_ci(y_true, y_score, method="bootstrap")


def test_auc_ci_small_n_boot_does_not_crash():
    # Arrange
    rng = np.random.default_rng(4)
    n = 30
    y_true = rng.integers(0, 2, size=n)
    y_score = y_true + rng.normal(0, 1, size=n)
    # Act
    result = auc_ci(y_true, y_score, method="bootstrap", n_boot=10, seed=42)
    # Assert
    assert result["ci_lower"] <= result["ci_upper"]
