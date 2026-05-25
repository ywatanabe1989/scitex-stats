#!/usr/bin/env python3
# Time-stamp: "2025-01-15 00:00:00 (ywatanabe)"
# File: examples/tests/categorical/demo_fisher.py
# ----------------------------------------

"""
Demo script for Fisher's exact test examples.

Run with: python examples/tests/categorical/demo_fisher.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from scitex_stats._logging import getLogger
from scitex_stats._utils._normalizers import force_dataframe
from scitex_stats.tests.categorical._test_fisher import test_fisher

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def main(args):
    """Run demonstration."""
    logger.info("=" * 70)
    logger.info("Fisher's Exact Test - Examples")
    logger.info("=" * 70)

    # Example 1: Small sample treatment study
    logger.info("\nExample 1: Small sample treatment study")
    logger.info("-" * 70)
    observed1 = [[8, 2], [1, 5]]  # Treatment: Success/Failure
    result1 = test_fisher(
        observed1,
        var_row="Treatment",
        var_col="Outcome",
        plot=True,
        verbose=True,
    )
    logger.info(force_dataframe(result1))
    plt.gcf().savefig("fisher_example1.jpg")
    plt.close("all")

    # Example 2: Case-control study (exposure × disease)
    logger.info("\nExample 2: Case-control study")
    logger.info("-" * 70)
    observed2 = [[12, 5], [8, 20]]  # Exposure: Cases/Controls
    result2 = test_fisher(
        observed2,
        var_row="Exposure",
        var_col="Disease",
        plot=True,
        verbose=True,
    )
    logger.info(force_dataframe(result2))
    plt.gcf().savefig("fisher_example2.jpg")
    plt.close("all")

    # Example 3: One-tailed test (expect positive association)
    logger.info("\nExample 3: One-tailed test (alternative='greater')")
    logger.info("-" * 70)
    observed3 = [[10, 2], [3, 8]]
    logger.info("Two-tailed:")
    test_fisher(observed3, alternative="two-sided", verbose=True)
    logger.info("\nOne-tailed (greater):")
    test_fisher(observed3, alternative="greater", verbose=True)

    # Example 4: Using pandas DataFrame with labels
    print("\nExample 4: Using pandas DataFrame")
    print("-" * 70)
    df4 = pd.DataFrame(
        [[15, 5], [3, 10]],
        index=["Group A", "Group B"],
        columns=["Success", "Failure"],
    )
    df4.index.name = "Group"
    df4.columns.name = "Outcome"
    result4 = test_fisher(df4, plot=True)
    print(force_dataframe(result4))
    plt.gcf().savefig("example4_dataframe.jpg")
    plt.close("all")

    # Example 5: Compare Fisher vs Chi-square
    print("\nExample 5: Compare Fisher's exact vs Chi-square")
    print("-" * 70)
    observed5 = [[5, 10], [10, 5]]
    fisher_result = test_fisher(observed5, plot=False)

    from scitex_stats.tests.categorical._test_chi2 import test_chi2

    chi2_result = test_chi2(observed5, plot=False)

    print(f"Fisher's exact test: p = {fisher_result['pvalue']:.4f} (exact)")
    print(f"Chi-square test:     p = {chi2_result['pvalue']:.4f} (approximation)")
    print("-> Fisher's exact provides exact p-value, chi-square is approximation")

    # Example 6: Very small sample
    print("\nExample 6: Very small sample (chi-square not recommended)")
    print("-" * 70)
    observed6 = [[2, 3], [1, 4]]
    result6 = test_fisher(observed6, var_row="Group", var_col="Response", plot=True)
    print(force_dataframe(result6))
    print("Fisher's exact test is ideal for small samples")
    plt.gcf().savefig("example6_small_sample.jpg")
    plt.close("all")

    # Example 7: Strong association
    print("\nExample 7: Strong positive association")
    print("-" * 70)
    observed7 = [[20, 2], [3, 18]]
    result7 = test_fisher(observed7, var_row="Factor A", var_col="Factor B", plot=True)
    print(force_dataframe(result7))
    print(f"Very strong association: OR = {result7['statistic']:.1f}")
    plt.gcf().savefig("example7_strong_association.jpg")
    plt.close("all")

    # Example 8: No association (OR ~ 1)
    print("\nExample 8: No association")
    print("-" * 70)
    observed8 = [[10, 10], [10, 10]]
    result8 = test_fisher(observed8, plot=True)
    print(force_dataframe(result8))
    print(f"OR = {result8['statistic']:.2f} ~ 1 (no association)")
    plt.gcf().savefig("example8_no_association.jpg")
    plt.close("all")

    # Example 9: Negative association (OR < 1)
    print("\nExample 9: Negative association (OR < 1)")
    print("-" * 70)
    observed9 = [[2, 15], [12, 8]]
    result9 = test_fisher(
        observed9, var_row="Treatment", var_col="Adverse Event", plot=True
    )
    print(force_dataframe(result9))
    print(f"OR = {result9['statistic']:.3f} < 1 (negative association)")
    plt.gcf().savefig("example9_negative_association.jpg")
    plt.close("all")

    # Example 10: Export to CSV
    print("\nExample 10: Export to CSV")
    print("-" * 70)
    result10 = test_fisher(
        observed2, var_row="Exposure", var_col="Disease", return_as="dataframe"
    )
    result10.to_csv("fisher_demo.csv", index=False)  # type: ignore[union-attr]
    # stx.io.save(result10, 'fisher_demo.tex') removed -- leaf-rule
    print("Exported to CSV")
    print(result10)

    logger.info(f"\n{'=' * 70}")
    logger.info("All examples completed")
    logger.info(f"{'=' * 70}")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Run main without the scitex umbrella session helpers."""
    import matplotlib

    matplotlib.use("Agg")

    args = parse_args()
    return main(args)


if __name__ == "__main__":
    run_main()

# EOF
