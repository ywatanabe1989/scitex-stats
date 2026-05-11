#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-01 20:48:08 (ywatanabe)"
# File: scitex_stats/correct/_correct_fdr.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Apply False Discovery Rate (FDR) correction for multiple comparisons
  - Implement Benjamini-Hochberg and Benjamini-Yekutieli procedures
  - Support both dict and DataFrame inputs
  - Maintain full result information with adjusted values

Dependencies:
  - packages: numpy, pandas

IO:
  - input: Test results with p-values (dict, list of dicts, or DataFrame)
  - output: Results with adjusted p-values and significance (same format as input)
"""

"""Imports"""
from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib
import matplotlib.axes
import numpy as np
import pandas as pd

from scitex_stats._logging import getLogger

logger = getLogger(__name__)

"""Functions"""


def correct_fdr(
    results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    alpha: float = 0.05,
    method: Literal["bh", "by"] = "bh",
    return_as: str = None,
    verbose: bool = True,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """
    Apply False Discovery Rate (FDR) correction for multiple comparisons.

    Parameters
    ----------
    results : dict, list of dict, or DataFrame
        Test results containing 'pvalue' field(s)
        - Single dict: one test result
        - List of dicts: multiple test results
        - DataFrame: multiple test results (one per row)
    alpha : float, default 0.05
        False discovery rate to control
    method : {'bh', 'by'}, default 'bh'
        FDR control method:
        - 'bh': Benjamini-Hochberg (assumes independence or positive dependence)
        - 'by': Benjamini-Yekutieli (valid under arbitrary dependence)
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
        - pvalue_adjusted: FDR-adjusted p-value (q-value)
        - alpha_adjusted: Effective alpha threshold for each test
        - rejected: Whether null hypothesis is rejected
        - pstars: Significance stars (using adjusted p-value)

    Notes
    -----
    **Benjamini-Hochberg (BH) Procedure:**

    For m tests with p-values p₁ ≤ p₂ ≤ ... ≤ pₘ:

    1. Order p-values from smallest to largest
    2. Find largest i such that: p_i ≤ (i/m) × α
    3. Reject H₀ for tests 1, 2, ..., i

    Adjusted p-values (q-values):

    .. math::
        q_i = \\min_{j \\geq i} \\left( \\frac{m \\cdot p_j}{j} \\right)

    **Benjamini-Yekutieli (BY) Procedure:**

    More conservative, valid under arbitrary dependence:

    .. math::
        q_i = c(m) \\cdot \\min_{j \\geq i} \\left( \\frac{m \\cdot p_j}{j} \\right)

    where :math:`c(m) = \\sum_{i=1}^{m} \\frac{1}{i} \\approx \\ln(m) + 0.5772`

    **FDR vs FWER:**

    - FWER (Bonferroni): Controls probability of ANY false positive
    - FDR: Controls expected proportion of false positives among rejections

    .. math::
        FDR = E\\left[\\frac{\\text{False Positives}}{\\text{Total Rejections}}\\right]

    **Advantages:**
    - More powerful than Bonferroni (especially with many tests)
    - Scales well to large m
    - Balances Type I and Type II errors

    **Disadvantages:**
    - Weaker control than FWER methods
    - May allow some false positives
    - BH requires independence assumption

    **When to use:**
    - Large number of tests (m > 10)
    - Exploratory analysis
    - Can tolerate some false positives
    - Need higher power than Bonferroni

    References
    ----------
    .. [1] Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing". Journal of the Royal Statistical Society, Series B, 57(1), 289-300.
    .. [2] Benjamini, Y., & Yekutieli, D. (2001). "The control of the false
           discovery rate in multiple testing under dependency". Annals of
           Statistics, 29(4), 1165-1188.

    Examples
    --------
    >>> # Multiple tests
    >>> results = [
    ...     {'pvalue': 0.001, 'var_x': 'A', 'var_y': 'B'},
    ...     {'pvalue': 0.010, 'var_x': 'A', 'var_y': 'C'},
    ...     {'pvalue': 0.050, 'var_x': 'B', 'var_y': 'C'},
    ...     {'pvalue': 0.100, 'var_x': 'A', 'var_y': 'D'}
    ... ]
    >>> corrected = correct_fdr(results)
    >>> [r['pvalue_adjusted'] for r in corrected]
    [0.004, 0.02, 0.0666..., 0.1]

    >>> # BH vs BY comparison
    >>> corrected_bh = correct_fdr(results, method='bh')
    >>> corrected_by = correct_fdr(results, method='by')
    >>> corrected_bh[0]['pvalue_adjusted'] < corrected_by[0]['pvalue_adjusted']
    True
    """
    from scitex_stats._utils._formatters import p2stars
    from scitex_stats._utils._normalizers import force_dataframe, to_dict

    if verbose:
        method_name = "Benjamini-Hochberg" if method == "bh" else "Benjamini-Yekutieli"
        logger.info(f"Applying FDR correction ({method_name})")

    # Store original input type
    input_type = type(results)
    is_single_dict = isinstance(results, dict)

    # Convert to DataFrame for processing
    if isinstance(results, dict):
        results_list = [results]
    elif isinstance(results, list):
        results_list = results
    else:  # DataFrame
        results_list = None

    if results_list is not None:
        df = force_dataframe(results_list, fill_na=False, enforce_types=False)
    else:
        df = results.copy()

    # Store original order
    df["_original_order"] = np.arange(len(df))

    # Number of tests
    m = len(df)
    if verbose:
        logger.info(f"Number of tests: {m}, alpha: {alpha}")

    # Sort by p-value
    df_sorted = df.sort_values("pvalue").copy()
    pvalues = df_sorted["pvalue"].values

    # Compute adjusted p-values (q-values)
    if method == "bh":
        # Benjamini-Hochberg
        ranks = np.arange(1, m + 1)
        q_values = np.minimum.accumulate((pvalues * m / ranks)[::-1])[::-1]
        q_values = np.minimum(q_values, 1.0)

    elif method == "by":
        # Benjamini-Yekutieli
        # c(m) = sum(1/i) for i in 1:m ≈ ln(m) + γ (Euler-Mascheroni constant)
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        ranks = np.arange(1, m + 1)
        q_values = np.minimum.accumulate((pvalues * m * c_m / ranks)[::-1])[::-1]
        q_values = np.minimum(q_values, 1.0)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bh' or 'by'.")

    df_sorted["pvalue_adjusted"] = q_values

    # Compute adjusted alpha threshold for each test
    if "alpha" in df.columns:
        alpha_values = df_sorted["alpha"].fillna(alpha)
    else:
        alpha_values = alpha

    # For FDR, the effective alpha threshold varies by rank
    ranks = np.arange(1, m + 1)
    if method == "bh":
        alpha_adj = alpha_values * ranks / m
    else:  # by
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        alpha_adj = alpha_values * ranks / (m * c_m)

    df_sorted["alpha_adjusted"] = alpha_adj

    # Determine rejections using BH/BY procedure
    # Find largest i where p_i <= (i/m) * alpha
    if method == "bh":
        threshold = alpha * ranks / m
    else:
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        threshold = alpha * ranks / (m * c_m)

    # Find the largest rank where p-value is below threshold
    significant = pvalues <= threshold
    if np.any(significant):
        max_significant_rank = np.max(np.where(significant)[0]) + 1
        df_sorted["rejected"] = np.arange(1, m + 1) <= max_significant_rank
    else:
        df_sorted["rejected"] = False

    # Update significance stars based on adjusted p-values
    df_sorted["pstars"] = df_sorted["pvalue_adjusted"].apply(p2stars)

    # Restore original order
    df_result = df_sorted.sort_values("_original_order").drop(
        columns=["_original_order"]
    )

    # Log results summary
    if verbose:
        rejections = df_result["rejected"].sum()
        logger.info(f"FDR correction complete: {rejections}/{m} hypotheses rejected")

        # Log detailed results if not too many tests
        if m <= 10:
            logger.info("\nDetailed results:")
            for idx, row in df_result.iterrows():
                comparison = ""
                if "var_x" in row and "var_y" in row:
                    comparison = f"{row['var_x']} vs {row['var_y']}: "
                elif "test_method" in row:
                    comparison = f"{row['test_method']}: "
                elif "comparison" in row:
                    comparison = f"{row['comparison']}: "

                logger.info(
                    f"  {comparison}"
                    f"p = {row['pvalue']:.4f} → q = {row['pvalue_adjusted']:.4f} "
                    f"{row['pstars']}, rejected = {row['rejected']}"
                )

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
        _plot_fdr(df_result, alpha, method, ax)

    # Determine return format
    if return_as == "dataframe":
        return df_result
    elif return_as == "dict":
        if is_single_dict:
            return to_dict(df_result, row=0)
        else:
            return df_result.to_dict("records")
    else:
        # Match input format
        if input_type == dict:
            return to_dict(df_result, row=0)
        elif input_type == list:
            return df_result.to_dict("records")
        else:  # DataFrame
            return df_result


def _plot_fdr(df, alpha, method, ax):
    """Create visualization for FDR correction on given axes."""
    m = len(df)
    x = np.arange(m)

    # Plot original p-values and q-values
    ax.scatter(x, df["pvalue"], label="Original p-values", alpha=0.7, s=100, color="C0")
    ax.scatter(
        x,
        df["pvalue_adjusted"],
        label="Q-values (FDR-adjusted)",
        alpha=0.7,
        s=100,
        color="C1",
        marker="s",
    )

    # Connect original to adjusted with lines
    for i in range(m):
        ax.plot(
            [i, i],
            [df["pvalue"].iloc[i], df["pvalue_adjusted"].iloc[i]],
            "k-",
            alpha=0.3,
            linewidth=0.5,
        )

    # Add significance threshold
    ax.axhline(
        alpha, color="red", linestyle="--", linewidth=2, alpha=0.5, label=f"α = {alpha}"
    )

    # Formatting
    method_name = "Benjamini-Hochberg" if method == "bh" else "Benjamini-Yekutieli"
    ax.set_xlabel("Test Index")
    ax.set_ylabel("P-value / Q-value")
    rejections = df["rejected"].sum()
    ax.set_title(
        f"FDR Correction ({method_name}, m={m} tests)\n"
        f"{rejections}/{m} hypotheses rejected"
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis labels if there are comparison names
    if m <= 20:
        labels = []
        for idx, row in df.iterrows():
            if "var_x" in row and "var_y" in row:
                labels.append(f"{row['var_x']}\nvs\n{row['var_y']}")
            elif "test_method" in row:
                labels.append(row["test_method"])
            elif "comparison" in row:
                labels.append(row["comparison"])
            else:
                labels.append(f"Test {len(labels) + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xlabel(f"Test Index (1-{m})")


# Demo lives in _demo_correct_fdr.py.

# EOF
