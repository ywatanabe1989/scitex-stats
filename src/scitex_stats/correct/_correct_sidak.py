#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-01 20:29:45 (ywatanabe)"
# File: scitex_stats/correct/_correct_sidak.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Apply Šidák correction for multiple comparisons
  - Adjust p-values and significance thresholds assuming independence
  - Support both dict and DataFrame inputs
  - More powerful than Bonferroni under independence assumption

Dependencies:
  - packages: numpy, pandas

IO:
  - input: Test results with p-values (dict, list of dicts, or DataFrame)
  - output: Results with adjusted p-values and significance (same format as input)
"""

"""Imports"""
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.axes
import numpy as np
import pandas as pd

from scitex_stats._logging import getLogger

logger = getLogger(__name__)

"""Functions"""


def correct_sidak(
    results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    alpha: float = 0.05,
    return_as: str = None,
    verbose: bool = True,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """
    Apply Šidák correction for multiple comparisons.

    Parameters
    ----------
    results : dict, list of dict, or DataFrame
        Test results containing 'pvalue' field(s)
    alpha : float, default 0.05
        Family-wise error rate (FWER) to control
    return_as : {'dict', 'dataframe', None}, optional
        Force specific return format. If None, matches input format.
    verbose : bool, default True
        Whether to log progress information
    plot : bool, default False
        Whether to generate visualization
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and plot=True, creates new figure.
        If provided, automatically enables plotting.

    Returns
    -------
    results : dict, list of dict, or DataFrame
        Results with added fields:
        - pvalue_adjusted: Šidák-adjusted p-value
        - alpha_adjusted: Šidák-adjusted alpha threshold
        - rejected: Whether null hypothesis is rejected (using adjusted values)
        - pstars: Significance stars (using adjusted p-value)

    Notes
    -----
    The Šidák correction is less conservative than Bonferroni and assumes
    independence between tests. It controls the family-wise error rate (FWER).

    For m tests with family-wise error rate α:

    .. math::
        \\alpha_{adj} = 1 - (1 - \\alpha)^{1/m}

    .. math::
        p_{adj,i} = 1 - (1 - p_i)^m

    See Also
    --------
    correct_bonferroni : More conservative alternative
    correct_holm : Sequential Bonferroni method
    correct_fdr : FDR control (less conservative)

    References
    ----------
    .. [1] Šidák, Z. (1967). "Rectangular Confidence Regions for the Means of
           Multivariate Normal Distributions". Journal of the American Statistical
           Association, 62(318), 626-633.
    """
    from scitex_stats._utils._formatters import p2stars

    if verbose:
        logger.info("Applying Šidák correction")

    # Determine input format
    single_result = False
    if isinstance(results, dict):
        results = [results]
        single_result = True
    elif isinstance(results, pd.DataFrame):
        results_list = results.to_dict("records")
        input_was_df = True
    else:
        results_list = results
        input_was_df = False

    if isinstance(results, list):
        results_list = results
        input_was_df = False

    # Number of tests
    m = len(results_list)
    if verbose:
        logger.info(f"Number of tests: {m}, alpha: {alpha}")

    # Šidák-adjusted alpha: α_adj = 1 - (1 - α)^(1/m)
    alpha_adj = 1.0 - (1.0 - alpha) ** (1.0 / m)

    # Apply correction to each result
    corrected_results = []
    for r in results_list:
        r_copy = r.copy()
        pval = r["pvalue"]

        # Adjusted p-value: p_adj = 1 - (1 - p)^m
        if pval >= 1.0:
            pval_adj = 1.0
        else:
            pval_adj = 1.0 - (1.0 - pval) ** m
            pval_adj = min(pval_adj, 1.0)

        r_copy["pvalue_adjusted"] = pval_adj
        r_copy["alpha_adjusted"] = alpha_adj
        r_copy["rejected"] = pval_adj < alpha
        r_copy["significant"] = r_copy["rejected"]
        r_copy["pstars"] = p2stars(pval_adj)

        corrected_results.append(r_copy)

    # Log results summary
    if verbose:
        rejections = sum(r["rejected"] for r in corrected_results)
        logger.info(f"Šidák correction complete: {rejections}/{m} hypotheses rejected")
        logger.info(f"Adjusted alpha threshold: {alpha_adj:.6f}")

        if m <= 10:
            logger.info("\nDetailed results:")
            for r in corrected_results:
                comparison = ""
                if "var_x" in r and "var_y" in r:
                    comparison = f"{r['var_x']} vs {r['var_y']}: "
                elif "test_method" in r:
                    comparison = f"{r['test_method']}: "
                elif "comparison" in r:
                    comparison = f"{r['comparison']}: "

                logger.info(
                    f"  {comparison}"
                    f"p = {r['pvalue']:.4f} → p_adj = {r['pvalue_adjusted']:.4f} "
                    f"{r['pstars']}, rejected = {r['rejected']}"
                )

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    if plot:
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
        _plot_sidak(corrected_results, alpha, alpha_adj, m, ax)

    # Format output
    if single_result:
        return corrected_results[0]

    if return_as == "dataframe" or (return_as is None and input_was_df):
        return pd.DataFrame(corrected_results)

    return corrected_results


def _plot_sidak(corrected_results, alpha, alpha_adj, m, ax):
    """Create visualization for Šidák correction on given axes."""
    x = np.arange(m)
    pvalues = [r["pvalue"] for r in corrected_results]
    pvalues_adj = [r["pvalue_adjusted"] for r in corrected_results]

    ax.scatter(x, pvalues, label="Original p-values", alpha=0.7, s=100, color="C0")
    ax.scatter(
        x,
        pvalues_adj,
        label="Adjusted p-values",
        alpha=0.7,
        s=100,
        color="C1",
        marker="s",
    )

    for i in range(m):
        ax.plot(
            [i, i],
            [pvalues[i], pvalues_adj[i]],
            "k-",
            alpha=0.3,
            linewidth=0.5,
        )

    ax.axhline(
        alpha,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"α = {alpha}",
    )
    ax.axhline(
        alpha_adj,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"α_adj = {alpha_adj:.4f}",
    )

    ax.set_xlabel("Test Index")
    ax.set_ylabel("P-value")
    rejections = sum(r["rejected"] for r in corrected_results)
    ax.set_title(
        f"Šidák Correction (m={m} tests)\n{rejections}/{m} hypotheses rejected"
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if m <= 20:
        labels = []
        for r in corrected_results:
            if "var_x" in r and "var_y" in r:
                labels.append(f"{r['var_x']}\nvs\n{r['var_y']}")
            elif "test_method" in r:
                labels.append(r["test_method"])
            elif "comparison" in r:
                labels.append(r["comparison"])
            else:
                labels.append(f"Test {len(labels) + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xlabel(f"Test Index (1-{m})")


# Demo lives in _demo_correct_sidak.py.

# EOF
