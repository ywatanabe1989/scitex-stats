#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-07-05 00:00:00 (ywatanabe)"
# File: scitex_stats/resampling/_delong.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Fast O(n log n) midrank computation (ties -> average rank)
  - DeLong (Sun & Xu 2014) structural components (V10 / V01) for a
    single ROC-AUC and for a correlated pair of ROC-AUCs
  - Analytic variance (single AUC) and covariance matrix (paired AUC)
    estimators shared by ``auc_ci`` and ``delta_auc_ci``

Dependencies:
  - packages: numpy

IO:
  - input: binary labels + continuous scores (arrays)
  - output: AUC point estimate(s) + variance / covariance (floats / arrays)

This module is internal (leading underscore) — not part of the public
API. It exists so the single-AUC path (``_auc_ci.py``) and the
two-AUC path (``_delta_auc_ci.py``) share exactly one implementation
of the DeLong midrank/covariance math instead of duplicating it.
"""

"""Imports"""
from typing import Tuple

import numpy as np

"""Functions"""


def _validate_binary_labels(y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate ``y_true`` is binary and return (pos_mask, neg_mask).

    Parameters
    ----------
    y_true : array-like
        Binary class labels. The larger of the two unique values is
        treated as the positive class (matches the common convention
        of {0, 1} labels with 1 = positive).

    Returns
    -------
    (np.ndarray, np.ndarray)
        Boolean masks (pos_mask, neg_mask) into ``y_true``.

    Raises
    ------
    ValueError
        If ``y_true`` does not contain exactly two distinct classes.
    """
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    if classes.size != 2:
        raise ValueError(
            "auc_ci/delta_auc_ci require y_true to contain exactly 2 "
            f"classes; got {classes.size} unique value(s): {classes.tolist()}"
        )
    pos_label = classes.max()
    pos_mask = y_true == pos_label
    neg_mask = ~pos_mask
    return pos_mask, neg_mask


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks (average rank for ties) of a 1-D array.

    O(n log n) via argsort, following the fast DeLong algorithm
    (Sun & Xu, 2014; Fawcett-style implementations in common use).

    Parameters
    ----------
    x : np.ndarray
        1-D array of scores.

    Returns
    -------
    np.ndarray
        Midranks (1-indexed), same shape as ``x``.

    Examples
    --------
    >>> _compute_midrank(np.array([1.0, 2.0, 2.0, 3.0]))
    array([1. , 2.5, 2.5, 4. ])
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    order = np.argsort(x)
    sorted_x = x[order]

    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # Average rank for the tied block [i, j] (1-indexed ranks)
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    midrank = np.empty(n, dtype=float)
    midrank[order] = ranks
    return midrank


def _structural_components(
    pos_scores: np.ndarray, neg_scores: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute AUC and DeLong structural components for one score set.

    Parameters
    ----------
    pos_scores : np.ndarray
        Scores for the positive-class samples, shape (n_pos,).
    neg_scores : np.ndarray
        Scores for the negative-class samples, shape (n_neg,).

    Returns
    -------
    auc : float
        Mann-Whitney U based AUC estimate.
    v01 : np.ndarray
        Structural component per positive sample, shape (n_pos,).
    v10 : np.ndarray
        Structural component per negative sample, shape (n_neg,).
    """
    n_pos = pos_scores.shape[0]
    n_neg = neg_scores.shape[0]

    tx = _compute_midrank(pos_scores)
    ty = _compute_midrank(neg_scores)
    tz = _compute_midrank(np.concatenate([pos_scores, neg_scores]))

    tz_pos = tz[:n_pos]
    tz_neg = tz[n_pos:]

    v01 = (tz_pos - tx) / n_neg
    v10 = 1.0 - (tz_neg - ty) / n_pos

    auc = tz_pos.sum() / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)

    return float(auc), v01, v10


def _delong_auc_var(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float, int, int]:
    """Analytic DeLong AUC + variance for a single ROC curve.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_score : array-like
        Continuous scores.

    Returns
    -------
    (auc, var, n_pos, n_neg) : tuple
        AUC point estimate, its DeLong variance estimate, and the
        positive/negative sample counts.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    pos_mask, neg_mask = _validate_binary_labels(y_true)

    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]
    n_pos, n_neg = pos_scores.shape[0], neg_scores.shape[0]

    auc, v01, v10 = _structural_components(pos_scores, neg_scores)

    var_v01 = np.var(v01, ddof=1) if n_pos > 1 else 0.0
    var_v10 = np.var(v10, ddof=1) if n_neg > 1 else 0.0

    var = var_v01 / n_pos + var_v10 / n_neg
    return auc, float(var), n_pos, n_neg


def _delong_two_auc_covar(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> Tuple[float, float, float, float, float, int, int]:
    """Analytic DeLong AUC + covariance for two correlated ROC curves.

    Both score sets are evaluated on the SAME samples/labels (paired
    design), so the full 2x2 DeLong covariance matrix is used —
    including the off-diagonal covariance term — rather than treating
    the two AUCs as independent.

    Parameters
    ----------
    y_true : array-like
        Binary labels shared by both score sets.
    score_a, score_b : array-like
        Continuous scores from the two competing classifiers.

    Returns
    -------
    (auc_a, auc_b, var_a, var_b, covar_ab, n_pos, n_neg) : tuple
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    pos_mask, neg_mask = _validate_binary_labels(y_true)

    auc_a, v01_a, v10_a = _structural_components(
        score_a[pos_mask], score_a[neg_mask]
    )
    auc_b, v01_b, v10_b = _structural_components(
        score_b[pos_mask], score_b[neg_mask]
    )
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos > 1:
        sx = np.cov(np.vstack([v01_a, v01_b]))  # 2x2, ddof=1 default
    else:
        sx = np.zeros((2, 2))
    if n_neg > 1:
        sy = np.cov(np.vstack([v10_a, v10_b]))
    else:
        sy = np.zeros((2, 2))

    s = sx / n_pos + sy / n_neg  # 2x2 covariance matrix of (auc_a, auc_b)

    var_a = float(s[0, 0])
    var_b = float(s[1, 1])
    covar_ab = float(s[0, 1])

    return auc_a, auc_b, var_a, var_b, covar_ab, n_pos, n_neg


# EOF
