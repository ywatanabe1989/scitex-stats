#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-07-05 00:00:00 (ywatanabe)"
# File: scitex_stats/auto/_recommend_overload.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Raw-vector convenience overload for `recommend_tests`: accepts
    two raw arrays (`x`, `y`) + a `paired` flag and auto-builds the
    `StatContext` internally, mirroring how `run_test` accepts raw
    arrays instead of requiring the caller to build context objects
    by hand.

Dependencies:
  - packages: numpy

IO:
  - input: two raw arrays, OR a pre-built StatContext (both accepted)
  - output: list of recommended test names (delegates entirely to the
    existing `scitex_stats.auto._selector.recommend_tests` — no
    recommendation logic is duplicated here)
"""

"""Imports"""
from typing import List, Optional, Union

import numpy as np

from scitex_stats.auto._context import StatContext, _infer_outcome_type
from scitex_stats.auto._rules import TestFamily
from scitex_stats.auto._selector import recommend_tests as _recommend_tests_from_ctx

"""Functions"""


def _build_context_from_raw_arrays(
    x: np.ndarray, y: np.ndarray, paired: bool
) -> StatContext:
    """Build a StatContext from two raw sample arrays.

    Infers sample sizes directly from the array lengths, the outcome
    type from the pooled data (reusing the same private helper
    StatContext.from_data() uses internally), and sets the design to
    "within" when paired else "between" — matching StatContext's own
    paired/design inference rules (see StatContext.__post_init__).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    pooled = np.concatenate([x, y])
    outcome_type = _infer_outcome_type(pooled)
    design = "within" if paired else "between"
    return StatContext(
        n_groups=2,
        sample_sizes=[int(x.shape[0]), int(y.shape[0])],
        outcome_type=outcome_type,
        design=design,
        paired=paired,
    )


def recommend_tests(
    ctx_or_x: Union[StatContext, np.ndarray],
    y: Optional[np.ndarray] = None,
    paired: bool = False,
    top_k: int = 3,
    families: Optional[List[TestFamily]] = None,
) -> List[str]:
    """Recommend tests for a StatContext OR two raw sample arrays.

    Two calling conventions:

    - ``recommend_tests(ctx, top_k=3)`` — the original interface;
      pass a pre-built :class:`StatContext`.
    - ``recommend_tests(x, y, paired=True, top_k=3)`` — convenience
      overload; pass two raw arrays and a ``paired`` flag, and the
      StatContext is built internally (sample sizes from array
      lengths, outcome type inferred from the pooled data, design set
      to "within"/"between" from ``paired``). Delegates entirely to
      the original context-based ``recommend_tests`` — no
      recommendation logic is duplicated.

    Parameters
    ----------
    ctx_or_x : StatContext or array-like
        Either a pre-built context, or the first raw sample array.
    y : array-like, optional
        Second raw sample array. Required (and only used) when
        ``ctx_or_x`` is not a :class:`StatContext`.
    paired : bool, default False
        Whether the two raw arrays are paired/repeated-measures.
        Only used in the raw-array calling convention.
    top_k : int, default 3
        Number of top tests to return.
    families : list of TestFamily or None
        Families to consider. If None, uses standard test families.

    Returns
    -------
    test_names : list of str
        Internal names of recommended tests, sorted by priority.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.normal(0, 1, 30)
    >>> y = np.random.normal(0.5, 1, 32)
    >>> recommended = recommend_tests(x, y, paired=False, top_k=3)
    >>> "brunner_munzel" in recommended
    True
    """
    if isinstance(ctx_or_x, StatContext):
        return _recommend_tests_from_ctx(ctx_or_x, top_k=top_k, families=families)

    if y is None:
        raise TypeError(
            "recommend_tests(x, y, ...) requires `y` when the first argument "
            "is not a StatContext"
        )

    ctx = _build_context_from_raw_arrays(ctx_or_x, y, paired)
    return _recommend_tests_from_ctx(ctx, top_k=top_k, families=families)


# EOF
