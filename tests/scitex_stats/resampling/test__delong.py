"""Tests for `scitex_stats.resampling._delong` (internal DeLong helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from scitex_stats.resampling._delong import (
    _compute_midrank,
    _delong_auc_var,
    _delong_two_auc_covar,
    _validate_binary_labels,
)


def test_compute_midrank_handles_ties():
    # Arrange
    x = np.array([1.0, 2.0, 2.0, 3.0])
    # Act
    ranks = _compute_midrank(x)
    # Assert
    assert np.allclose(ranks, [1.0, 2.5, 2.5, 4.0])


def test_compute_midrank_no_ties_matches_plain_ranks():
    # Arrange
    x = np.array([3.0, 1.0, 2.0])
    # Act
    ranks = _compute_midrank(x)
    # Assert
    assert np.allclose(ranks, [3.0, 1.0, 2.0])


def test_validate_binary_labels_raises_on_single_class():
    # Arrange
    y_true = np.array([1, 1, 1, 1])
    # Act
    # (call happens inside the Assert block below)
    # Assert
    with pytest.raises(ValueError, match="2 classes"):
        _validate_binary_labels(y_true)


def test_validate_binary_labels_pos_mask_matches_larger_class():
    # Arrange
    y_true = np.array([0, 1, 0, 1])
    # Act
    pos_mask, _ = _validate_binary_labels(y_true)
    # Assert
    assert np.array_equal(pos_mask, [False, True, False, True])


def test_validate_binary_labels_neg_mask_matches_smaller_class():
    # Arrange
    y_true = np.array([0, 1, 0, 1])
    # Act
    _, neg_mask = _validate_binary_labels(y_true)
    # Assert
    assert np.array_equal(neg_mask, [True, False, True, False])


def test_delong_auc_var_perfect_separation_gives_auc_one():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    # Act
    auc, _, _, _ = _delong_auc_var(y_true, y_score)
    # Assert
    assert np.isclose(auc, 1.0)


def test_delong_auc_var_perfect_separation_gives_nonnegative_var():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    # Act
    _, var, _, _ = _delong_auc_var(y_true, y_score)
    # Assert
    assert var >= 0


def test_delong_auc_var_perfect_separation_counts_samples_correctly():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    # Act
    _, _, n_pos, n_neg = _delong_auc_var(y_true, y_score)
    # Assert
    assert (n_pos, n_neg) == (2, 2)


def test_delong_auc_var_identical_scores_give_auc_half():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    # Act
    auc, _, _, _ = _delong_auc_var(y_true, y_score)
    # Assert
    assert np.isclose(auc, 0.5)


def test_delong_auc_var_identical_scores_give_finite_var():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    # Act
    _, var, _, _ = _delong_auc_var(y_true, y_score)
    # Assert
    assert np.isfinite(var)


def test_delong_auc_var_identical_scores_give_nonnegative_var():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    # Act
    _, var, _, _ = _delong_auc_var(y_true, y_score)
    # Assert
    assert var >= 0


def test_delong_two_auc_covar_auc_a_in_unit_interval():
    # Arrange
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 0.5, size=n)
    score_b = y_true + rng.normal(0, 0.5, size=n)
    # Act
    auc_a, _, _, _, _, _, _ = _delong_two_auc_covar(y_true, score_a, score_b)
    # Assert
    assert 0.0 <= auc_a <= 1.0


def test_delong_two_auc_covar_auc_b_in_unit_interval():
    # Arrange
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 0.5, size=n)
    score_b = y_true + rng.normal(0, 0.5, size=n)
    # Act
    _, auc_b, _, _, _, _, _ = _delong_two_auc_covar(y_true, score_a, score_b)
    # Assert
    assert 0.0 <= auc_b <= 1.0


def test_delong_two_auc_covar_var_a_nonnegative():
    # Arrange
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 0.5, size=n)
    score_b = y_true + rng.normal(0, 0.5, size=n)
    # Act
    _, _, var_a, _, _, _, _ = _delong_two_auc_covar(y_true, score_a, score_b)
    # Assert
    assert var_a >= 0


def test_delong_two_auc_covar_var_b_nonnegative():
    # Arrange
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 0.5, size=n)
    score_b = y_true + rng.normal(0, 0.5, size=n)
    # Act
    _, _, _, var_b, _, _, _ = _delong_two_auc_covar(y_true, score_a, score_b)
    # Assert
    assert var_b >= 0


def test_delong_two_auc_covar_sample_counts_sum_to_n():
    # Arrange
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 0.5, size=n)
    score_b = y_true + rng.normal(0, 0.5, size=n)
    # Act
    _, _, _, _, _, n_pos, n_neg = _delong_two_auc_covar(y_true, score_a, score_b)
    # Assert
    assert n_pos + n_neg == n
