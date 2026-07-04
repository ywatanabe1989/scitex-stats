#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-07-05 00:00:00 (ywatanabe)"
# File: scitex_stats/effect_sizes/_effect_size_from_ci.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Standardized effect size derived from a point estimate + its
    confidence interval, for metrics that only expose a CI/SE (e.g. a
    bootstrap or DeLong CI) rather than raw samples to feed cohens_d.

Dependencies:
  - packages: scipy

IO:
  - input: point estimate + CI bounds (floats)
  - output: standardized effect size (float)
"""

"""Imports"""
from scipy.stats import norm

from scitex_stats.effect_sizes._cohens_d import interpret_cohens_d

"""Functions"""


def effect_size_from_ci(
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    ci: float = 95,
) -> float:
    """Standardized effect size from a point estimate and its CI.

    Many metrics (bootstrap AUC, DeLong-derived deltas, meta-analytic
    summary estimates, ...) only expose a point estimate plus a
    confidence interval — no raw per-sample data to hand to
    :func:`cohens_d`. This computes a Cohen's-d-like standardized
    effect size directly from that CI, back-deriving the standard
    error via the normal approximation:

    .. math::
        SE = \\frac{CI_{upper} - CI_{lower}}{2z}

        d = \\frac{estimate}{SE}

    where :math:`z` is the normal quantile for the requested
    confidence level (``scipy.stats.norm.ppf``, e.g. ``z ≈ 1.96`` for
    a 95% CI — so ``2z ≈ 3.92`` matches the commonly hand-rolled
    constant, but generalizes correctly to any ``ci``).

    Parameters
    ----------
    estimate : float
        Point estimate (e.g. a mean difference, an AUC delta).
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    ci : float, default 95
        Confidence level in percent that ``ci_lower``/``ci_upper``
        correspond to.

    Returns
    -------
    float
        Standardized effect size (``estimate / SE``).

    Raises
    ------
    ValueError
        If ``ci_upper <= ci_lower`` (degenerate or reversed interval).

    Examples
    --------
    >>> effect_size_from_ci(0.5, 0.3, 0.7)  # 95% CI, half-width 0.2
    2.45...
    """
    if ci_upper <= ci_lower:
        raise ValueError(
            f"ci_upper ({ci_upper}) must be greater than ci_lower ({ci_lower})"
        )

    z = norm.ppf(0.5 + ci / 200.0)
    se = (ci_upper - ci_lower) / (2.0 * z)
    return float(estimate / se)


def interpret_effect_size_from_ci(d: float) -> str:
    """Interpret a CI-derived standardized effect size.

    Reuses the standard Cohen's d interpretation thresholds
    (Cohen, 1988) since ``effect_size_from_ci`` produces a value on
    the same standardized scale (estimate divided by its own SE).

    Parameters
    ----------
    d : float
        Standardized effect size from :func:`effect_size_from_ci`.

    Returns
    -------
    str
        Interpretation string ("negligible", "small", "medium", "large").

    Examples
    --------
    >>> interpret_effect_size_from_ci(0.3)
    'small'
    """
    return interpret_cohens_d(d)


# EOF
