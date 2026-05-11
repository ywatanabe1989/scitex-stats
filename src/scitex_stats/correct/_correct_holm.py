#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-01 21:00:01 (ywatanabe)"
# File: scitex_stats/correct/_correct_holm.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Perform Holm-Bonferroni correction for multiple comparisons
  - More powerful than standard Bonferroni while controlling FWER
  - Sequential rejection procedure
  - Support dict, list, or DataFrame inputs

Dependencies:
  - packages: numpy, pandas

IO:
  - input: Test results (dict, list of dicts, or DataFrame)
  - output: Corrected results with adjusted p-values
"""

"""Imports"""
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.axes
import numpy as np
import pandas as pd

from scitex_stats._logging import getLogger

logger = getLogger(__name__)

"""Functions"""


def correct_holm(
    results: Union[Dict, List[Dict], pd.DataFrame],
    alpha: float = 0.05,
    verbose: bool = True,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> Union[List[Dict], pd.DataFrame]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    results : dict, list of dicts, or DataFrame
        Statistical test results containing 'pvalue' field
    alpha : float, default 0.05
        Family-wise error rate (FWER)
    verbose : bool, default True
        Whether to log progress information
    plot : bool, default False
        Whether to generate visualization
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and plot=True, creates new figure.
        If provided, automatically enables plotting.

    Returns
    -------
    corrected_results : list of dicts or DataFrame
        Results with added fields:
        - pvalue_adjusted: Adjusted p-value
        - alpha_adjusted: Adjusted alpha threshold (for reference)
        - rejected: Whether null hypothesis is rejected after correction

    Notes
    -----
    The Holm-Bonferroni method (Holm, 1979) is a sequentially rejective
    multiple testing procedure that controls the family-wise error rate (FWER).
    It is uniformly more powerful than the standard Bonferroni correction.

    **Procedure**:
    1. Order p-values from smallest to largest: p₁ ≤ p₂ ≤ ... ≤ pₘ
    2. For each i = 1, 2, ..., m:
       - Compare pᵢ with α/(m - i + 1)
       - Reject H₀ᵢ if pᵢ ≤ α/(m - i + 1)
       - Stop at the first i where pᵢ > α/(m - i + 1)
       - Reject all H₀₁, ..., H₀ᵢ₋₁; accept all others

    **Adjusted p-values**:
    For reporting, adjusted p-values are computed as:

    .. math::
        \\tilde{p}_i = \\max_{j \\leq i} \\{(m - j + 1) p_j\\}

    Ensuring monotonicity: p̃₁ ≤ p̃₂ ≤ ... ≤ p̃ₘ

    **Advantages over Bonferroni**:
    - More powerful (detects more true positives)
    - Still controls FWER at level α
    - Simple step-down procedure
    - No independence assumption required

    **When to use**:
    - Multiple pairwise comparisons (e.g., post-hoc tests after ANOVA)
    - Want stronger control than FDR but more power than Bonferroni
    - Number of tests is moderate (m < 100)

    **Comparison with other methods**:
    - **Bonferroni**: More conservative, less powerful
    - **FDR (Benjamini-Hochberg)**: More powerful, controls different error rate
    - **Šidák**: Similar to Bonferroni, assumes independence

    References
    ----------
    .. [1] Holm, S. (1979). "A simple sequentially rejective multiple test
           procedure". Scandinavian Journal of Statistics, 6(2), 65-70.
    .. [2] Aickin, M., & Gensler, H. (1996). "Adjusting for multiple testing
           when reporting research results: the Bonferroni vs Holm methods".
           American Journal of Public Health, 86(5), 726-728.

    Examples
    --------
    >>> # Single test result
    >>> result = {'pvalue': 0.01, 'test_method': 'test'}
    >>> corrected = correct_holm(result)
    >>> corrected[0]['pvalue_adjusted']
    0.01

    >>> # Multiple tests
    >>> results = [
    ...     {'pvalue': 0.001, 'test_method': 't-test'},
    ...     {'pvalue': 0.04, 'test_method': 't-test'},
    ...     {'pvalue': 0.03, 'test_method': 't-test'}
    ... ]
    >>> corrected = correct_holm(results, alpha=0.05)
    >>> [r['rejected'] for r in corrected]
    [True, False, True]

    >>> # As DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame(results)
    >>> df_corrected = correct_holm(df)
    """
    from scitex_stats._utils._normalizers import force_dataframe

    if verbose:
        logger.info("Applying Holm-Bonferroni correction")

    # Convert to list of dicts if needed
    return_as_dataframe = isinstance(results, pd.DataFrame)

    if isinstance(results, dict):
        results = [results]
    elif isinstance(results, pd.DataFrame):
        results = results.to_dict("records")

    # Extract p-values
    if not results:
        raise ValueError("Empty results provided")

    if "pvalue" not in results[0]:
        raise ValueError("Results must contain 'pvalue' field")

    m = len(results)
    if verbose:
        logger.info(f"Number of tests: {m}, alpha: {alpha}")

    # Create indexed results for tracking original order
    indexed_results = [(i, r) for i, r in enumerate(results)]

    # Sort by p-value (ascending)
    sorted_results = sorted(indexed_results, key=lambda x: x[1]["pvalue"])

    # Compute adjusted p-values using Holm's method
    adjusted_pvalues = []
    for i, (orig_idx, result) in enumerate(sorted_results):
        p = result["pvalue"]

        # Holm adjustment: p_adj = max over j≤i of (m - j + 1) * p_j
        # This ensures monotonicity
        adj_p = (m - i) * p  # Initial adjustment

        # Enforce monotonicity: adjusted p-values must be non-decreasing
        if i > 0:
            adj_p = max(adj_p, adjusted_pvalues[i - 1])

        # Cap at 1.0
        adj_p = min(adj_p, 1.0)

        adjusted_pvalues.append(adj_p)

    # Apply corrections to results
    corrected_results = []
    for i, (orig_idx, result) in enumerate(sorted_results):
        corrected = result.copy()
        corrected["pvalue_adjusted"] = round(adjusted_pvalues[i], 6)
        corrected["alpha_adjusted"] = round(alpha / (m - i), 6)  # For reference
        corrected["rejected"] = adjusted_pvalues[i] <= alpha

        # Add original index for restoration
        corrected["_orig_idx"] = orig_idx

        corrected_results.append(corrected)

    # Restore original order
    corrected_results.sort(key=lambda x: x["_orig_idx"])

    # Remove temporary index field
    for r in corrected_results:
        del r["_orig_idx"]

    # Log results summary
    if verbose:
        rejections = sum(r["rejected"] for r in corrected_results)
        logger.info(f"Holm correction complete: {rejections}/{m} hypotheses rejected")
        logger.info(f"Adjusted alpha range: {alpha / m:.6f} to {alpha:.6f}")

        # Log detailed results if not too many tests
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
                    f"p = {r['pvalue']:.4f} → p_adj = {r['pvalue_adjusted']:.4f}, "
                    f"rejected = {r['rejected']}"
                )

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
        _plot_holm(corrected_results, alpha, m, ax)

    # Convert to DataFrame if input was DataFrame
    if return_as_dataframe:
        return force_dataframe(corrected_results)

    return corrected_results


def _plot_holm(corrected_results, alpha, m, ax):
    """Create visualization for Holm correction on given axes."""
    x = np.arange(m)
    pvalues = [r["pvalue"] for r in corrected_results]
    pvalues_adj = [r["pvalue_adjusted"] for r in corrected_results]

    # Plot original and adjusted p-values
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

    # Connect original to adjusted with lines
    for i in range(m):
        ax.plot(
            [i, i],
            [pvalues[i], pvalues_adj[i]],
            "k-",
            alpha=0.3,
            linewidth=0.5,
        )

    # Add significance thresholds
    ax.axhline(
        alpha,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"α = {alpha}",
    )
    ax.axhline(
        alpha / m,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"α_min = {alpha / m:.4f}",
    )

    # Formatting
    ax.set_xlabel("Test Index")
    ax.set_ylabel("P-value")
    rejections = sum(r["rejected"] for r in corrected_results)
    ax.set_title(f"Holm Correction (m={m} tests)\n{rejections}/{m} hypotheses rejected")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis labels if there are comparison names
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



# Demo lives in _demo_correct_holm.py.

# EOF
