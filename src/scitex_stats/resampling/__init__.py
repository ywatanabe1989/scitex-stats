#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resampling-based confidence intervals for statistical estimates.

This module provides:

- ``auc_ci``: confidence interval for a single ROC-AUC, via analytic
  DeLong (Sun & Xu 2014) variance or percentile bootstrap.
- ``delta_auc_ci``: confidence interval + significance test for the
  difference of two CORRELATED ROC-AUCs (same samples/labels), using
  the full DeLong covariance matrix (accounts for the correlation
  between the two classifiers' scores).
- ``bootstrap_ci``: generic percentile bootstrap CI for any metric
  function of one or more equally-sized, paired-resampled arrays.
- ``paired_auc_effect_size``: standardized (Cohen's-d-like) effect
  size for the difference of two correlated AUCs, reusing the same
  DeLong covariance machinery as ``delta_auc_ci``.
"""

from ._auc_ci import auc_ci
from ._bootstrap_ci import bootstrap_ci
from ._delta_auc_ci import delta_auc_ci, paired_auc_effect_size

__all__ = [
    "auc_ci",
    "delta_auc_ci",
    "bootstrap_ci",
    "paired_auc_effect_size",
]
