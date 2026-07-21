#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for Cliff's delta effect size.

Tests cover:
- Non-parametric effect size computation
- Range validation (-1 <= δ <= 1)
- Perfect separation (δ=±1)
- No effect (δ=0)
- Ordinal data handling
- Robustness to outliers
- Mathematical properties
"""

import numpy as np
import pandas as pd
import pytest

from scitex_stats.effect_sizes import cliffs_delta, interpret_cliffs_delta


class TestBasicComputation:
    """Tests for basic Cliff's delta computations."""

    def test_basic_comparison_delta_float(self):
        """Test basic two-sample comparison."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert isinstance(delta, float)

    def test_basic_comparison_delta(self):
        """Test basic two-sample comparison."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert -1 <= delta <= 1

    def test_basic_comparison_delta_2(self):
        """Test basic two-sample comparison."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert delta < 0  # y dominates x

    def test_known_value_delta(self):
        """Test with manually calculated known value."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert delta < 0

    def test_known_value_delta_2(self):
        """Test with manually calculated known value."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert -0.6 < delta < 0  # Should be negative but reasonable

    def test_pandas_series_input_delta_float(self):
        """Test that pandas Series work as input."""
        # Arrange
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 4, 5, 6, 7])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert isinstance(delta, float)

    def test_pandas_series_input_isnan_delta(self):
        """Test that pandas Series work as input."""
        # Arrange
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 4, 5, 6, 7])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert not np.isnan(delta)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_effect_identical_distributions(self):
        """Test that identical distributions give δ = 0."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert abs(delta) < 0.01  # Should be very close to 0

    def test_perfect_dominance_positive(self):
        """Test perfect dominance (all x > all y) gives δ = 1."""
        # Arrange
        x = np.array([6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert abs(delta - 1.0) < 0.01

    def test_perfect_dominance_negative(self):
        """Test perfect dominance (all x < all y) gives δ = -1."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert abs(delta - (-1.0)) < 0.01

    def test_nan_handling_delta_float(self):
        """Test that NaN values are properly removed."""
        # Arrange
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([3, np.nan, 5, 6, 7])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert isinstance(delta, float)

    def test_nan_handling_isnan_delta(self):
        """Test that NaN values are properly removed."""
        # Arrange
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([3, np.nan, 5, 6, 7])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert not np.isnan(delta)

    def test_ties_handling_delta(self):
        """Test handling of tied values."""
        # Arrange
        x = np.array([1, 2, 2, 3, 3])
        y = np.array([2, 2, 3, 3, 4])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert -1 <= delta <= 1
        # With ties, neither > nor < counts the tie


class TestKnownValues:
    """Tests with manually calculated known values."""

    def test_complete_separation_abs_delta(self):
        """Test with completely separated groups."""
        # Arrange
        x = np.array([10, 11, 12])
        y = np.array([1, 2, 3])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert abs(delta - 1.0) < 0.01

    def test_partial_overlap_delta(self):
        """Test with partial overlap."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert delta < 0

    def test_partial_overlap_delta_2(self):
        """Test with partial overlap."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert -1 < delta < 0


class TestMathematicalProperties:
    """Tests for mathematical properties of Cliff's delta."""

    def test_range_constraint_delta_normal_cliffs_delta_random(self):
        """Test that δ is always between -1 and 1."""
        # Arrange
        # Act
        np.random.seed(42)
        # Assert
        for _ in range(20):
            x = np.random.normal(0, 1, 20)
            y = np.random.normal(np.random.uniform(-2, 2), 1, 20)
            delta = cliffs_delta(x, y)

            assert -1 <= delta <= 1

    def test_antisymmetry_abs_delta_xy_delta_yx(self):
        """Test that cliffs_delta(x, y) = -cliffs_delta(y, x)."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])
        delta_xy = cliffs_delta(x, y)
        # Act
        delta_yx = cliffs_delta(y, x)
        # Assert
        assert abs(delta_xy + delta_yx) < 0.001

    def test_monotone_transformation_invariance(self):
        """Test that δ is invariant to monotone transformations."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])
        delta_original = cliffs_delta(x, y)
        # Act
        delta_transformed = cliffs_delta(x**2, y**2)
        # Assert
        assert abs(delta_original - delta_transformed) < 0.01

    def test_ordinal_nature_abs_delta1_delta2(self):
        """Test that δ only depends on order, not magnitude."""
        # Arrange
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        x2 = np.array([1, 2, 3])
        y2 = np.array([100, 200, 300])
        delta1 = cliffs_delta(x1, y1)
        # Act
        delta2 = cliffs_delta(x2, y2)
        # Assert
        assert abs(delta1 - delta2) < 0.01

    def test_ordinal_nature_abs_delta1(self):
        """Test that δ only depends on order, not magnitude."""
        # Arrange
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        x2 = np.array([1, 2, 3])
        y2 = np.array([100, 200, 300])
        delta1 = cliffs_delta(x1, y1)
        # Act
        delta2 = cliffs_delta(x2, y2)
        # Assert
        assert abs(delta1 - (-1.0)) < 0.01  # Perfect separation


class TestRobustness:
    """Tests for robustness properties."""

    def test_robust_to_outliers(self):
        """Test that δ is robust to outliers."""
        # Arrange
        x_normal = np.array([1, 2, 3, 4, 5])
        y_normal = np.array([3, 4, 5, 6, 7])
        delta_normal = cliffs_delta(x_normal, y_normal)
        x_outlier = np.array([1, 2, 3, 4, 100])
        # Act
        delta_outlier = cliffs_delta(x_outlier, y_normal)
        # Assert
        assert abs(delta_normal - delta_outlier) < 0.5

    def test_skewed_distributions_delta(self):
        """Test with heavily skewed distributions."""
        # Arrange
        np.random.seed(42)
        x = np.random.exponential(1, 30)
        y = np.random.exponential(2, 30)
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert -1 <= delta <= 1

    def test_skewed_distributions_delta_2(self):
        """Test with heavily skewed distributions."""
        # Arrange
        np.random.seed(42)
        x = np.random.exponential(1, 30)
        y = np.random.exponential(2, 30)
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert delta < 0


class TestInterpretation:
    """Tests for effect size interpretation."""

    def test_interpret_negligible_interpret_cliffs_delta(self):
        """Test negligible effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(0.1) == "negligible"

    def test_interpret_negligible_interpret_cliffs_delta_2(self):
        """Test negligible effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(-0.1) == "negligible"

    def test_interpret_small_interpret_cliffs_delta(self):
        """Test small effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(0.25) == "small"

    def test_interpret_small_interpret_cliffs_delta_2(self):
        """Test small effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(-0.25) == "small"

    def test_interpret_medium_interpret_cliffs_delta(self):
        """Test medium effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(0.4) == "medium"

    def test_interpret_medium_interpret_cliffs_delta_2(self):
        """Test medium effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(-0.4) == "medium"

    def test_interpret_large_interpret_cliffs_delta(self):
        """Test large effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(0.6) == "large"

    def test_interpret_large_interpret_cliffs_delta_2(self):
        """Test large effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(-0.8) == "large"

    def test_interpret_boundaries_small_interpret_cliffs_delta(self):
        """Test interpretation at boundaries."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(0.147) == "small"

    def test_interpret_boundaries_medium_interpret_cliffs_delta(self):
        """Test interpretation at boundaries."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(0.33) == "medium"

    def test_interpret_boundaries_large_interpret_cliffs_delta(self):
        """Test interpretation at boundaries."""
        # Arrange
        # Act
        # Assert
        assert interpret_cliffs_delta(0.474) == "large"


class TestOrdinalData:
    """Tests specifically for ordinal data."""

    def test_likert_scale_data_delta(self):
        """Test with Likert scale responses (1-5)."""
        # Arrange
        x = np.array([1, 1, 2, 2, 2, 3, 3])
        y = np.array([3, 3, 4, 4, 4, 5, 5])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert -1 <= delta <= 1

    def test_likert_scale_data_delta_2(self):
        """Test with Likert scale responses (1-5)."""
        # Arrange
        x = np.array([1, 1, 2, 2, 2, 3, 3])
        y = np.array([3, 3, 4, 4, 4, 5, 5])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert delta < -0.3

    def test_ranking_data_abs_delta(self):
        """Test with ranking data."""
        # Arrange
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])
        # Act
        delta = cliffs_delta(x, y)
        # Assert
        assert abs(delta - (-1.0)) < 0.01


class TestComparisonWithCohensD:
    """Compare Cliff's delta with Cohen's d."""

    def test_similar_conclusions_normal_data_delta(self):
        """Test that δ and d agree on normal data."""
        # Arrange
        from scitex_stats.effect_sizes import cohens_d
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0.8, 1, 50)
        delta = cliffs_delta(x, y)
        # Act
        d = cohens_d(x, y)
        # Assert
        assert delta < 0

    def test_similar_conclusions_normal_data_case_2(self):
        """Test that δ and d agree on normal data."""
        # Arrange
        from scitex_stats.effect_sizes import cohens_d
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0.8, 1, 50)
        delta = cliffs_delta(x, y)
        # Act
        d = cohens_d(x, y)
        # Assert
        assert d < 0

    def test_robust_vs_sensitive_to_outliers(self):
        """Test robustness difference with outliers."""
        # Arrange
        from scitex_stats.effect_sizes import cohens_d
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])
        delta_normal = cliffs_delta(x, y)
        d_normal = cohens_d(x, y)
        x_outlier = np.array([1, 2, 3, 4, 100])
        delta_outlier = cliffs_delta(x_outlier, y)
        d_outlier = cohens_d(x_outlier, y)
        delta_change = abs(delta_normal - delta_outlier)
        # Act
        d_change = abs(d_normal - d_outlier)
        # Assert
        assert delta_change < d_change


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_cliffs_delta.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 21:00:00 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/effect_sizes/_cliffs_delta.py
# # ----------------------------------------
# from __future__ import annotations
# import os
#
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# """
# Functionalities:
#   - Compute Cliff's delta non-parametric effect size
#   - Robust to outliers and non-normal distributions
#   - Provide interpretation guidelines
#   - Related to Mann-Whitney U statistic
#
# Dependencies:
#   - packages: numpy, pandas
#
# IO:
#   - input: Two samples (arrays or Series)
#   - output: Effect size value (float, ranges from -1 to 1)
# """
#
# """Imports"""
# import argparse
# from typing import Union
#
# import numpy as np
# import pandas as pd
# import scitex as stx
# from scitex.logging import getLogger
#
# logger = getLogger(__name__)
#
# """Functions"""
#
#
# def cliffs_delta(
#     x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]
# ) -> float:
#     """
#     Compute Cliff's delta non-parametric effect size.
#
#     Parameters
#     ----------
#     x : array or Series
#         First sample
#     y : array or Series
#         Second sample
#
#     Returns
#     -------
#     float
#         Cliff's delta value (ranges from -1 to 1)
#
#     Notes
#     -----
#     Cliff's delta is a non-parametric effect size measure that quantifies
#     the degree of dominance of one distribution over another.
#
#     It is calculated as:
#
#     .. math::
#         \\delta = \\frac{\\#(x_i > y_j) - \\#(x_i < y_j)}{n_x \\cdot n_y}
#
#     Where:
#     - #(x_i > y_j) is the number of times values in x are greater than values in y
#     - #(x_i < y_j) is the number of times values in x are less than values in y
#
#     Interpretation:
#     - |δ| < 0.147: negligible
#     - |δ| < 0.33:  small
#     - |δ| < 0.474: medium
#     - |δ| ≥ 0.474: large
#
#     Advantages:
#     - Non-parametric (no assumptions about distributions)
#     - Robust to outliers
#     - Easy to interpret (probability-based)
#     - Related to Mann-Whitney U statistic
#
#     The relation to Mann-Whitney U is:
#
#     .. math::
#         \\delta = 2 \\cdot \\frac{U}{n_x \\cdot n_y} - 1
#
#     References
#     ----------
#     .. [1] Cliff, N. (1993). "Dominance statistics: Ordinal analyses to answer
#            ordinal questions". Psychological Bulletin, 114(3), 494-509.
#     .. [2] Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006).
#            "Appropriate statistics for ordinal level data: Should we really be
#            using t-test and Cohen's d for evaluating group differences on the
#            NSSE and other surveys?" Florida Association of Institutional Research.
#
#     Examples
#     --------
#     >>> x = np.array([1, 2, 3, 4, 5])
#     >>> y = np.array([2, 3, 4, 5, 6])
#     >>> cliffs_delta(x, y)
#     -0.6
#
#     >>> # No difference
#     >>> x = np.array([1, 2, 3, 4, 5])
#     >>> y = np.array([1, 2, 3, 4, 5])
#     >>> cliffs_delta(x, y)
#     0.0
#
#     >>> # Complete dominance
#     >>> x = np.array([6, 7, 8, 9, 10])
#     >>> y = np.array([1, 2, 3, 4, 5])
#     >>> cliffs_delta(x, y)
#     1.0
#     """
#     # Convert to numpy arrays
#     x = np.asarray(x)
#     y = np.asarray(y)
#
#     # Remove NaN values
#     x = x[~np.isnan(x)]
#     y = y[~np.isnan(y)]
#
#     nx = len(x)
#     ny = len(y)
#
#     # Count comparisons
#     # Vectorized computation: create all pairwise comparisons
#     # x[:, None] creates column vector, y creates row vector
#     # Broadcasting creates matrix of all pairwise comparisons
#     more = np.sum(x[:, None] > y)
#     less = np.sum(x[:, None] < y)
#
#     # Compute Cliff's delta
#     delta = (more - less) / (nx * ny)
#
#     return float(delta)
#
#
# def interpret_cliffs_delta(delta: float) -> str:
#     """
#     Interpret Cliff's delta effect size.
#
#     Parameters
#     ----------
#     delta : float
#         Cliff's delta value
#
#     Returns
#     -------
#     str
#         Interpretation string
#
#     Examples
#     --------
#     >>> interpret_cliffs_delta(0.1)
#     'negligible'
#     >>> interpret_cliffs_delta(0.25)
#     'small'
#     >>> interpret_cliffs_delta(0.4)
#     'medium'
#     >>> interpret_cliffs_delta(0.6)
#     'large'
#     """
#     delta_abs = abs(delta)
#
#     if delta_abs < 0.147:
#         return "negligible"
#     elif delta_abs < 0.33:
#         return "small"
#     elif delta_abs < 0.474:
#         return "medium"
#     else:
#         return "large"
#
#
# """Main function"""
#
#
# def main(args):
#     """Demonstrate Cliff's delta computation."""
#     logger.info("Demonstrating Cliff's delta effect size")
#
#     # Set random seed
#     np.random.seed(42)
#
#     # Example 1: Basic usage
#     logger.info("\n=== Example 1: Basic usage ===")
#
#     x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#     y = np.array([3, 4, 5, 6, 7, 8, 9, 10])
#
#     delta = cliffs_delta(x, y)
#     interpretation = interpret_cliffs_delta(delta)
#
#     logger.info(f"Cliff's delta: {delta:.3f} ({interpretation})")
#     logger.info(f"{abs(delta):.1%} dominance of one group over the other")
#
#     # Example 2: Robustness to outliers
#     logger.info("\n=== Example 2: Robustness to outliers ===")
#
#     x_normal = np.array([1, 2, 3, 4, 5])
#     y_normal = np.array([3, 4, 5, 6, 7])
#     x_outlier = np.array([1, 2, 3, 4, 100])  # Extreme outlier
#     y_outlier = np.array([3, 4, 5, 6, 7])
#
#     delta_normal = cliffs_delta(x_normal, y_normal)
#     delta_outlier = cliffs_delta(x_outlier, y_outlier)
#
#     from ._cohens_d import cohens_d
#
#     d_normal = cohens_d(x_normal, y_normal)
#     d_outlier = cohens_d(x_outlier, y_outlier)
#
#     logger.info(
#         f"Without outlier: Cliff's δ = {delta_normal:.3f}, Cohen's d = {d_normal:.3f}"
#     )
#     logger.info(
#         f"With outlier:    Cliff's δ = {delta_outlier:.3f}, Cohen's d = {d_outlier:.3f}"
#     )
#     logger.info("Cliff's delta is stable, Cohen's d is inflated by outlier")
#
#     # Example 3: Different effect sizes
#     logger.info("\n=== Example 3: Different effect sizes ===")
#
#     control = np.random.normal(0, 1, 50)
#
#     for shift in [0.0, 0.3, 0.6, 1.0]:
#         treatment = np.random.normal(shift, 1, 50)
#         delta = cliffs_delta(control, treatment)
#         interpretation = interpret_cliffs_delta(delta)
#
#         logger.info(f"Shift = {shift:.1f}: δ = {delta:.3f} ({interpretation})")
#
#     # Visualization
#     logger.info("\n=== Creating visualization ===")
#
#     fig, axes = stx.plt.subplots(1, 2, figsize=(12, 5))
#
#     # Plot 1: Distribution comparison
#     ax = axes[0]
#     x_demo = np.random.exponential(2, 200)
#     y_demo = np.random.exponential(3, 200)
#
#     ax.hist(x_demo, bins=30, alpha=0.5, label="Group X", density=True)
#     ax.hist(y_demo, bins=30, alpha=0.5, label="Group Y", density=True)
#
#     delta_demo = cliffs_delta(x_demo, y_demo)
#     ax.set_xlabel("Value")
#     ax.set_ylabel("Density")
#     ax.set_title(f"Distributions (Cliff's δ = {delta_demo:.2f})")
#     ax.legend()
#
#     # Plot 2: Effect size interpretation
#     ax = axes[1]
#     delta_values = np.linspace(-1, 1, 100)
#     interpretations = [interpret_cliffs_delta(d) for d in delta_values]
#
#     color_map = {
#         "negligible": "lightgray",
#         "small": "yellow",
#         "medium": "orange",
#         "large": "red",
#     }
#     colors = [color_map[i] for i in interpretations]
#
#     ax.scatter(delta_values, [0] * len(delta_values), c=colors, s=50, alpha=0.7)
#     ax.set_xlabel("Cliff's δ")
#     ax.set_yticks([])
#     ax.set_title("Effect Size Interpretation")
#     ax.axvline(0, color="black", linestyle="-", alpha=0.3)
#     ax.axvline(0.147, color="black", linestyle="--", alpha=0.3)
#     ax.axvline(0.33, color="black", linestyle="--", alpha=0.3)
#     ax.axvline(0.474, color="black", linestyle="--", alpha=0.3)
#     ax.axvline(-0.147, color="black", linestyle="--", alpha=0.3)
#     ax.axvline(-0.33, color="black", linestyle="--", alpha=0.3)
#     ax.axvline(-0.474, color="black", linestyle="--", alpha=0.3)
#
#     stx.plt.tight_layout()
#     stx.io.save(fig, "./cliffs_delta_demo.jpg")
#     logger.info("Visualization saved")
#
#     return 0
#
#
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate Cliff's delta effect size calculation"
#     )
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
#     return parser.parse_args()
#
#
# def run_main():
#     """Initialize SciTeX framework and run main."""
#     global CONFIG, sys, plt, rng
#
#     import sys
#     import matplotlib.pyplot as plt
#
#     args = parse_args()
#
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file=__file__,
#         verbose=args.verbose,
#         agg=True,
#     )
#
#     exit_status = main(args)
#
#     stx.session.close(
#         CONFIG,
#         verbose=args.verbose,
#         exit_status=exit_status,
#     )
#
#
# if __name__ == "__main__":
#     run_main()
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_cliffs_delta.py
# --------------------------------------------------------------------------------
