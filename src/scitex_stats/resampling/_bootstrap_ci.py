#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-07-05 00:00:00 (ywatanabe)"
# File: scitex_stats/resampling/_bootstrap_ci.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Generic percentile bootstrap confidence interval for any metric
    function of one or more equally-sized arrays
  - Paired resampling: the SAME random index array is applied to
    every array on each bootstrap iteration

Dependencies:
  - packages: numpy

IO:
  - input: a callable ``fn`` plus 1+ same-length arrays
  - output: normalized result dict (estimate, ci_lower, ci_upper, ...)
"""

"""Imports"""
from typing import Any, Callable, Dict

import numpy as np

"""Functions"""


def bootstrap_ci(
    fn: Callable[..., float],
    *arrays,
    n_boot: int = 2000,
    ci: float = 95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generic percentile bootstrap confidence interval.

    Resamples ``*arrays`` with replacement, using ONE shared random
    index array per iteration (so multiple arrays are resampled in a
    paired fashion — e.g. ``y_true``/``y_pred`` stay aligned), applies
    ``fn`` to each resample, and reports the percentile interval of
    the resulting bootstrap distribution.

    Parameters
    ----------
    fn : callable
        Metric function taking the same number of positional array
        arguments as passed in ``*arrays`` and returning a scalar.
    *arrays : array-like
        One or more arrays of equal length. Resampled in a paired
        fashion (same indices across all arrays, per iteration).
    n_boot : int, default 2000
        Number of bootstrap resamples.
    ci : float, default 95
        Confidence level in percent (e.g. 95 for a 95% CI).
    seed : int, default 42
        Seed for the random number generator (reproducibility).

    Returns
    -------
    dict
        ``estimate`` (fn applied to the original, unresampled data),
        ``ci_lower``, ``ci_upper``, ``ci``, ``n_boot``, ``formatted``.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = bootstrap_ci(np.mean, data, n_boot=500, seed=0)
    >>> result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
    True
    """
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0]
    for a in arrays:
        if a.shape[0] != n:
            raise ValueError("All arrays passed to bootstrap_ci must have the same length")

    estimate = float(fn(*arrays))

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [a[idx] for a in arrays]
        boot_stats[i] = fn(*resampled)

    alpha = (100.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_stats, alpha))
    ci_upper = float(np.percentile(boot_stats, 100.0 - alpha))

    formatted = f"Estimate = {estimate:.3f}, {ci:g}% CI [{ci_lower:.3f}, {ci_upper:.3f}]"

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci": ci,
        "n_boot": n_boot,
        "formatted": formatted,
    }


# EOF
