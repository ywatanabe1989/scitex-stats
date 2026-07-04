#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-07-05 00:00:00 (ywatanabe)"
# File: scitex_stats/resampling/_delta_auc_ci.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Confidence interval + significance test for the DIFFERENCE of two
    CORRELATED ROC-AUCs computed on the SAME samples/labels (e.g. two
    competing models scored on the same held-out set) — the classic
    DeLong test for comparing two correlated ROC curves
  - Full DeLong covariance matrix (including the off-diagonal
    covariance term) is used, so the resulting CI correctly narrows
    (or widens) relative to naively subtracting two independent CIs
  - Alternative percentile-bootstrap estimator (paired resampling)

Dependencies:
  - packages: numpy, scipy

IO:
  - input: binary labels + two continuous score arrays (paired)
  - output: normalized result dict (delta_auc, ci_lower, ci_upper, ...)
"""

"""Imports"""
from typing import Any, Dict, Literal

import numpy as np
from scipy.stats import norm

from scitex_stats.resampling._auc_ci import _auc_point_estimate
from scitex_stats.resampling._bootstrap_ci import bootstrap_ci
from scitex_stats.resampling._delong import (
    _delong_two_auc_covar,
    _validate_binary_labels,
)

"""Functions"""


def _delta_auc_point_estimate(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> float:
    """``auc(y_true, score_a) - auc(y_true, score_b)`` for one resample."""
    return _auc_point_estimate(y_true, score_a) - _auc_point_estimate(y_true, score_b)


def delta_auc_ci(
    y_true,
    score_a,
    score_b,
    ci: float = 95,
    method: Literal["delong", "bootstrap"] = "delong",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Confidence interval for the difference of two correlated AUCs.

    Implements the classic DeLong (1988) / Sun & Xu (2014) paired test
    for comparing two ROC curves evaluated on the SAME samples: the
    full 2x2 DeLong covariance matrix between the two AUCs is used so
    the correlation between the two classifiers' scores is correctly
    accounted for (``delta_var = var_a + var_b - 2*covar_ab``), rather
    than naively summing two independent variances.

    Parameters
    ----------
    y_true : array-like
        Binary class labels shared by both score arrays.
    score_a, score_b : array-like
        Continuous scores from the two competing classifiers,
        evaluated on the same samples as ``y_true``.
    ci : float, default 95
        Confidence level in percent.
    method : {"delong", "bootstrap"}, default "delong"
        - ``"delong"``: analytic DeLong covariance estimator.
        - ``"bootstrap"``: paired percentile bootstrap (the same
          resampled indices are applied to ``y_true``, ``score_a``,
          and ``score_b`` on every iteration).
    n_boot : int, default 2000
        Number of bootstrap resamples (only used when
        ``method="bootstrap"``).
    seed : int, default 42
        Random seed for the bootstrap method (reproducibility).

    Returns
    -------
    dict
        ``delta_auc`` (auc_a - auc_b), ``ci_lower``, ``ci_upper``,
        ``auc_a``, ``auc_b``, ``p_value``, ``method``, ``ci``,
        ``formatted``.

        For ``method="bootstrap"``, ``p_value`` is a percentile-based
        two-sided approximation
        (``2 * min(P(delta<=0), P(delta>=0))``, clipped to [0, 1])
        computed from the bootstrap distribution of the delta — this
        is an approximation, not an exact test, unlike the DeLong
        z-statistic used for ``method="delong"``.

    Raises
    ------
    ValueError
        If ``y_true`` does not contain exactly two classes, or if the
        three input arrays are not all the same length.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 0, 1, 1, 1])
    >>> score_a = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])  # perfect
    >>> score_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # chance
    >>> result = delta_auc_ci(y_true, score_a, score_b, method="delong")
    >>> result["delta_auc"] > 0
    True
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if not (y_true.shape[0] == score_a.shape[0] == score_b.shape[0]):
        raise ValueError("y_true, score_a, and score_b must all have the same length")
    _validate_binary_labels(y_true)

    if method == "delong":
        (
            auc_a,
            auc_b,
            var_a,
            var_b,
            covar_ab,
            n_pos,
            n_neg,
        ) = _delong_two_auc_covar(y_true, score_a, score_b)
        delta = auc_a - auc_b
        delta_var = max(var_a + var_b - 2.0 * covar_ab, 0.0)
        se = float(np.sqrt(delta_var))

        z_ci = norm.ppf(0.5 + ci / 200.0)
        ci_lower = float(np.clip(delta - z_ci * se, -1.0, 1.0))
        ci_upper = float(np.clip(delta + z_ci * se, -1.0, 1.0))

        if se > 0:
            z_stat = delta / se
            p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        else:
            p_value = 1.0 if delta == 0 else 0.0

    elif method == "bootstrap":
        auc_a = _auc_point_estimate(y_true, score_a)
        auc_b = _auc_point_estimate(y_true, score_b)
        delta = auc_a - auc_b

        rng = np.random.default_rng(seed)
        n = y_true.shape[0]
        boot_deltas = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_deltas[i] = _delta_auc_point_estimate(
                y_true[idx], score_a[idx], score_b[idx]
            )

        alpha = (100.0 - ci) / 2.0
        ci_lower = float(np.percentile(boot_deltas, alpha))
        ci_upper = float(np.percentile(boot_deltas, 100.0 - alpha))

        p_ge = float(np.mean(boot_deltas >= 0))
        p_le = float(np.mean(boot_deltas <= 0))
        p_value = float(np.clip(2.0 * min(p_ge, p_le), 0.0, 1.0))
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'delong' or 'bootstrap'.")

    formatted = (
        f"ΔAUC = {delta:.2f} (A={auc_a:.2f}, B={auc_b:.2f}), "
        f"{ci:g}% CI [{ci_lower:.2f}, {ci_upper:.2f}], p={p_value:.3g}"
    )

    return {
        "delta_auc": delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "p_value": p_value,
        "method": method,
        "ci": ci,
        "formatted": formatted,
    }


def paired_auc_effect_size(y_true, score_a, score_b) -> Dict[str, Any]:
    """Standardized effect size for two AUCs on the SAME samples.

    A matched-classifier analogue of Cohen's d: expresses the
    difference of two correlated AUCs (evaluated on the same
    samples/labels) as ``delta_auc / SE``, where ``SE`` is derived
    from the full DeLong covariance matrix — i.e. it reuses the exact
    same covariance machinery as :func:`delta_auc_ci` (via
    :func:`scitex_stats.resampling._delong._delong_two_auc_covar`)
    rather than re-deriving the DeLong variance/covariance terms.

    Parameters
    ----------
    y_true : array-like
        Binary class labels shared by both score arrays.
    score_a, score_b : array-like
        Continuous scores from the two competing classifiers,
        evaluated on the same samples as ``y_true``.

    Returns
    -------
    dict
        ``effect_size`` (``delta_auc / se``, 0.0 if ``se`` is 0),
        ``delta_auc``, ``se``, ``auc_a``, ``auc_b``, ``formatted``.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 0, 1, 1, 1])
    >>> score_a = np.array([0.1, 0.2, 0.35, 0.6, 0.7, 0.9])  # perfect
    >>> score_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # chance
    >>> result = paired_auc_effect_size(y_true, score_a, score_b)
    >>> result["effect_size"] > 0
    True
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if not (y_true.shape[0] == score_a.shape[0] == score_b.shape[0]):
        raise ValueError("y_true, score_a, and score_b must all have the same length")
    _validate_binary_labels(y_true)

    auc_a, auc_b, var_a, var_b, covar_ab, _, _ = _delong_two_auc_covar(
        y_true, score_a, score_b
    )
    delta = auc_a - auc_b
    var_delta = max(var_a + var_b - 2.0 * covar_ab, 0.0)
    se = float(np.sqrt(var_delta))
    effect_size = float(delta / se) if se > 0 else 0.0

    formatted = (
        f"Paired AUC effect size = {effect_size:.2f} "
        f"(ΔAUC={delta:.2f}, SE={se:.3f})"
    )

    return {
        "effect_size": effect_size,
        "delta_auc": delta,
        "se": se,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "formatted": formatted,
    }


# EOF
