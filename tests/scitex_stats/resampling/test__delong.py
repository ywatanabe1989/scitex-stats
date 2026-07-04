"""Tests for `scitex_stats.resampling._delong` (internal DeLong helpers)."""

from __future__ import annotations

import numpy as np

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
    # Act / Assert
    try:
        _validate_binary_labels(y_true)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "2 classes" in str(e) or "unique" in str(e)


def test_validate_binary_labels_returns_masks():
    # Arrange
    y_true = np.array([0, 1, 0, 1])
    # Act
    pos_mask, neg_mask = _validate_binary_labels(y_true)
    # Assert
    assert np.array_equal(pos_mask, [False, True, False, True])
    assert np.array_equal(neg_mask, [True, False, True, False])


def test_delong_auc_var_perfect_separation_gives_auc_one():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    # Act
    auc, var, n_pos, n_neg = _delong_auc_var(y_true, y_score)
    # Assert
    assert np.isclose(auc, 1.0)
    assert var >= 0
    assert n_pos == 2 and n_neg == 2


def test_delong_auc_var_identical_scores_give_auc_half():
    # Arrange
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    # Act
    auc, var, n_pos, n_neg = _delong_auc_var(y_true, y_score)
    # Assert
    assert np.isclose(auc, 0.5)
    assert np.isfinite(var)
    assert var >= 0


def test_delong_two_auc_covar_returns_valid_covariance_matrix_entries():
    # Arrange
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.integers(0, 2, size=n)
    score_a = y_true + rng.normal(0, 0.5, size=n)
    score_b = y_true + rng.normal(0, 0.5, size=n)
    # Act
    auc_a, auc_b, var_a, var_b, covar_ab, n_pos, n_neg = _delong_two_auc_covar(
        y_true, score_a, score_b
    )
    # Assert
    assert 0.0 <= auc_a <= 1.0
    assert 0.0 <= auc_b <= 1.0
    assert var_a >= 0
    assert var_b >= 0
    assert n_pos + n_neg == n
