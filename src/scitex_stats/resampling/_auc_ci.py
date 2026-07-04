#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-07-05 00:00:00 (ywatanabe)"
# File: scitex_stats/resampling/_auc_ci.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - 95% (or arbitrary) confidence interval for a single ROC-AUC
  - Two interchangeable estimators: analytic DeLong (Sun & Xu 2014)
    and percentile bootstrap

Dependencies:
  - packages: numpy, scipy

IO:
  - input: binary labels + continuous scores
  - output: normalized result dict (auc, ci_lower, ci_upper, ...)
"""

"""Imports"""
from typing import Any, Dict, Literal

import numpy as np
from scipy.stats import norm

from scitex_stats.resampling._bootstrap_ci import bootstrap_ci
from scitex_stats.resampling._delong import _delong_auc_var, _validate_binary_labels

"""Functions"""


def _auc_point_estimate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC point estimate — no sklearn.

    Degenerate single-class inputs (as can occur inside a bootstrap
    resample) return 0.5 by convention (no discriminative information
    to estimate from that particular resample) rather than raising,
    so a bootstrap loop can proceed uninterrupted.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    classes = np.unique(y_true)
    if classes.size != 2:
        return 0.5

    pos_mask = y_true == classes.max()
    neg_mask = ~pos_mask
    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]

    from scipy.stats import rankdata

    n_pos, n_neg = pos_scores.shape[0], neg_scores.shape[0]
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = rankdata(all_scores)
    rank_sum_pos = ranks[:n_pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auc_ci(
    y_true,
    y_score,
    ci: float = 95,
    method: Literal["delong", "bootstrap"] = "delong",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Confidence interval for a single ROC-AUC.

    Parameters
    ----------
    y_true : array-like
        Binary class labels.
    y_score : array-like
        Continuous scores / predicted probabilities.
    ci : float, default 95
        Confidence level in percent.
    method : {"delong", "bootstrap"}, default "delong"
        - ``"delong"``: analytic DeLong variance estimator (fast
          O(n log n) midrank algorithm, Sun & Xu 2014).
        - ``"bootstrap"``: percentile bootstrap over paired
          (y_true, y_score) resamples.
    n_boot : int, default 2000
        Number of bootstrap resamples (only used when
        ``method="bootstrap"``).
    seed : int, default 42
        Random seed for the bootstrap method (reproducibility).

    Returns
    -------
    dict
        ``auc``, ``ci_lower``, ``ci_upper``, ``ci``, ``method``,
        ``n_pos``, ``n_neg``, ``formatted``.

    Raises
    ------
    ValueError
        If ``y_true`` does not contain exactly two classes.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 0, 1, 1, 1])
    >>> y_score = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])
    >>> result = auc_ci(y_true, y_score, method="delong")
    >>> result["auc"]
    1.0
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    # Validate up-front (raises a clear ValueError for degenerate input)
    # before any DeLong/bootstrap work is attempted.
    _validate_binary_labels(y_true)

    if method == "delong":
        auc, var, n_pos, n_neg = _delong_auc_var(y_true, y_score)
        se = float(np.sqrt(max(var, 0.0)))
        z = norm.ppf(0.5 + ci / 200.0)
        ci_lower = float(np.clip(auc - z * se, 0.0, 1.0))
        ci_upper = float(np.clip(auc + z * se, 0.0, 1.0))
    elif method == "bootstrap":
        auc = _auc_point_estimate(y_true, y_score)
        boot_result = bootstrap_ci(
            _auc_point_estimate, y_true, y_score, n_boot=n_boot, ci=ci, seed=seed
        )
        ci_lower = boot_result["ci_lower"]
        ci_upper = boot_result["ci_upper"]
        pos_mask, neg_mask = _validate_binary_labels(y_true)
        n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'delong' or 'bootstrap'.")

    formatted = f"AUC = {auc:.2f}, {ci:g}% CI [{ci_lower:.2f}, {ci_upper:.2f}]"

    return {
        "auc": auc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci": ci,
        "method": method,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "formatted": formatted,
    }


# EOF
