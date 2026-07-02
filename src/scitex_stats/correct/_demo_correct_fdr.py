#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex_stats/correct/_demo_correct_fdr.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Demo script for False Discovery Rate (FDR) correction.

Run with: python -m scitex_stats.correct._demo_correct_fdr
"""

"""Imports"""
import argparse

import numpy as np
from scitex_dev import try_import_optional

stx = try_import_optional("scitex")

from scitex_stats._logging import getLogger

from ._correct_bonferroni import correct_bonferroni
from ._correct_fdr_ import correct_fdr

logger = getLogger(__name__)


def main(args):
    """Demonstrate FDR correction."""
    logger.info("Demonstrating False Discovery Rate correction")

    # Example 1: Single test (no correction needed)
    logger.info("\n=== Example 1: Single test ===")

    single_result = {
        "var_x": "Control",
        "var_y": "Treatment",
        "pvalue": 0.04,
        "alpha": 0.05,
    }
    correct_fdr(single_result, verbose=args.verbose)

    # Example 2: Multiple tests - BH method
    logger.info("\n=== Example 2: Multiple tests (Benjamini-Hochberg) ===")

    multiple_results = [
        {"var_x": "A", "var_y": "B", "pvalue": 0.001},
        {"var_x": "A", "var_y": "C", "pvalue": 0.010},
        {"var_x": "A", "var_y": "D", "pvalue": 0.050},
        {"var_x": "B", "var_y": "C", "pvalue": 0.100},
        {"var_x": "B", "var_y": "D", "pvalue": 0.200},
    ]

    corrected_bh = correct_fdr(
        multiple_results, method="bh", alpha=0.05, verbose=args.verbose
    )

    # Example 3: BH vs BY comparison
    logger.info("\n=== Example 3: BH vs BY comparison ===")

    corrected_by = correct_fdr(
        multiple_results, method="by", alpha=0.05, verbose=args.verbose
    )

    # Example 4: Bonferroni vs FDR
    logger.info("\n=== Example 4: Bonferroni vs FDR comparison ===")

    corrected_bonf = correct_bonferroni(multiple_results, alpha=0.05, verbose=False)

    n_rejected_bonf = sum(r["rejected"] for r in corrected_bonf)
    n_rejected_fdr = sum(r["rejected"] for r in corrected_bh)

    logger.info(f"Bonferroni rejections: {n_rejected_bonf}")
    logger.info(f"FDR (BH) rejections:   {n_rejected_fdr}")
    logger.info("FDR is more powerful (rejects more tests)")

    # Example 5: Many tests
    logger.info("\n=== Example 5: Large scale comparison (m=100) ===")

    np.random.seed(42)

    # Simulate 100 tests: 20 true positives, 80 true negatives
    many_results = []
    for i in range(20):
        p = np.random.beta(1, 50)  # Small p-values
        many_results.append(
            {
                "var_x": f"Var_{i}",
                "var_y": "Control",
                "pvalue": p,
                "truth": "positive",
            }
        )
    for i in range(20, 100):
        p = np.random.uniform(0.1, 1.0)  # Large p-values
        many_results.append(
            {
                "var_x": f"Var_{i}",
                "var_y": "Control",
                "pvalue": p,
                "truth": "negative",
            }
        )

    corrected_fdr_many = correct_fdr(many_results, method="bh", verbose=False)
    corrected_bonf_many = correct_bonferroni(many_results, verbose=False)

    def calc_metrics(corrected, truth_col="truth"):
        tp = sum(
            1 for r in corrected if r["rejected"] and r.get(truth_col) == "positive"
        )
        fp = sum(
            1 for r in corrected if r["rejected"] and r.get(truth_col) == "negative"
        )
        fn = sum(
            1 for r in corrected if not r["rejected"] and r.get(truth_col) == "positive"
        )
        tn = sum(
            1 for r in corrected if not r["rejected"] and r.get(truth_col) == "negative"
        )
        return tp, fp, fn, tn

    tp_fdr, fp_fdr, _, _ = calc_metrics(corrected_fdr_many)
    tp_bonf, fp_bonf, _, _ = calc_metrics(corrected_bonf_many)

    logger.info("FDR (BH) Performance:")
    logger.info(f"  True Positives:  {tp_fdr} / 20")
    logger.info(f"  False Positives: {fp_fdr}")
    logger.info(f"  Power: {tp_fdr / 20:.2%}")
    if tp_fdr + fp_fdr > 0:
        logger.info(f"  FDR: {fp_fdr / (tp_fdr + fp_fdr):.2%}")

    logger.info("\nBonferroni Performance:")
    logger.info(f"  True Positives:  {tp_bonf} / 20")
    logger.info(f"  False Positives: {fp_bonf}")
    logger.info(f"  Power: {tp_bonf / 20:.2%}")
    if tp_bonf + fp_bonf > 0:
        logger.info(f"  FDR: {fp_bonf / (tp_bonf + fp_bonf):.2%}")

    # Create visualization
    logger.info("\n=== Creating visualization ===")

    fig, axes = stx.plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Q-values vs P-values
    ax = axes[0, 0]
    test_pvalues = np.array([r["pvalue"] for r in corrected_bh])
    test_qvalues = np.array([r["pvalue_adjusted"] for r in corrected_bh])
    ax.scatter(test_pvalues, test_qvalues, s=100, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Original P-value")
    ax.set_ylabel("Adjusted P-value (Q-value)")
    ax.set_title("FDR: P-values vs Q-values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: BH vs BY
    ax = axes[0, 1]
    bh_qvalues = np.array([r["pvalue_adjusted"] for r in corrected_bh])
    by_qvalues = np.array([r["pvalue_adjusted"] for r in corrected_by])
    ax.scatter(bh_qvalues, by_qvalues, s=100, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("BH Q-value")
    ax.set_ylabel("BY Q-value")
    ax.set_title("BH vs BY (BY is more conservative)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Power comparison across m
    ax = axes[1, 0]
    m_vals = np.arange(5, 101, 5)
    alpha = 0.05
    alpha_bonf = alpha / m_vals
    alpha_fdr = alpha
    ax.plot(m_vals, alpha_bonf, label="Bonferroni", linewidth=2)
    ax.axhline(alpha_fdr, color="green", linestyle="--", linewidth=2, label="FDR (BH)")
    ax.set_xlabel("Number of Tests (m)")
    ax.set_ylabel("Effective α")
    ax.set_title("FDR Maintains Power vs Bonferroni")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Plot 4: ROC-like comparison
    ax = axes[1, 1]
    alphas = [0.001, 0.01, 0.05, 0.10, 0.20]
    bonf_tps, bonf_fps, fdr_tps, fdr_fps = [], [], [], []
    for a in alphas:
        corr_bonf = correct_bonferroni(many_results, alpha=a, verbose=False)
        corr_fdr = correct_fdr(many_results, alpha=a, method="bh", verbose=False)
        tp_b, fp_b, _, _ = calc_metrics(corr_bonf)
        tp_f, fp_f, _, _ = calc_metrics(corr_fdr)
        bonf_tps.append(tp_b / 20)
        bonf_fps.append(fp_b / 80)
        fdr_tps.append(tp_f / 20)
        fdr_fps.append(fp_f / 80)
    ax.plot(bonf_fps, bonf_tps, "o-", linewidth=2, markersize=8, label="Bonferroni")
    ax.plot(fdr_fps, fdr_tps, "s-", linewidth=2, markersize=8, label="FDR (BH)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Power)")
    ax.set_title("Power vs FPR Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()  # type: ignore[name-defined]

    stx.io.save(fig, "./fdr_demo.jpg")
    logger.info("Visualization saved")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate FDR correction")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    global CONFIG, sys, plt, rng

    import sys

    import matplotlib.pyplot as plt

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(  # type: ignore[union-attr]
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=args.verbose,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(  # type: ignore[union-attr]
        CONFIG,
        verbose=args.verbose,
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
