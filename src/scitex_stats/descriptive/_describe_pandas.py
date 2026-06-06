#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""describe_pandas — pandas-flavored descriptive statistics.

Ported from scitex-gen ``misc.py``. The legacy symbol name was just
``describe``, but ``scitex_stats.descriptive.describe`` already exists
with a much richer torch/numpy signature; to avoid a silent breakage
the legacy helper is exposed here as ``describe_pandas`` (preserving
the original behavior — pandas DataFrame input, mean_std / mean_ci /
median_iqr summaries returned as a dict).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def describe_pandas(df, method: str = "mean_std", round_factor: int = 3, axis: int = 0) -> dict:
    """Compute pandas-style descriptive statistics for a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        Input data; coerced to ``pandas.DataFrame``.
    method : {'mean_std', 'mean_ci', 'median_iqr'}, default 'mean_std'
        Summary statistic to compute.
    round_factor : int, default 3
        Decimal places for rounding the spread statistic in the
        ``median_iqr`` branch (and the count column).
    axis : int, default 0
        Axis along which to compute statistics.

    Returns
    -------
    dict
        ``{'n': ..., 'mean': ..., 'std' / 'ci': ...}`` for the
        mean variants, or ``{'n': ..., 'median': ..., 'iqr': ...}``
        for ``median_iqr``.

    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
    >>> result = describe_pandas(data, method='mean_std')
    >>> sorted(result)
    ['mean', 'n', 'std']
    """
    assert method in ("mean_std", "mean_ci", "median_iqr"), (
        f"method must be one of mean_std/mean_ci/median_iqr, got {method!r}"
    )
    df = pd.DataFrame(df)
    nn = df.notna().sum(axis=axis)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method in ("mean_std", "mean_ci"):
            mm = np.nanmean(df, axis=axis)
            if method == "mean_std":
                ss = np.nanstd(df, axis=axis)
                key = "std"
            else:  # mean_ci
                ss = 1.96 * np.nanstd(df, axis=axis) / np.sqrt(nn)
                key = "ci"
            return {
                "n": np.round(nn, 3),
                "mean": np.round(mm, 3),
                key: np.round(ss, 3),
            }
        # median_iqr
        med = df.median(axis=axis)
        iqr = df.quantile(0.75, axis=axis) - df.quantile(0.25, axis=axis)
        return {
            "n": np.round(nn, round_factor),
            "median": np.round(med, round_factor),
            "iqr": np.round(iqr, round_factor),
        }
