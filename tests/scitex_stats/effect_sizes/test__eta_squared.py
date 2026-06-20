#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for eta-squared effect size.

Tests cover:
- Basic ANOVA computations
- Multiple groups
- Range validation (0 <= η² <= 1)
- Perfect separation (η²=1)
- No effect (η²=0)
- Known values validation
- Mathematical properties
"""

import numpy as np
import pandas as pd
import pytest

from scitex_stats.effect_sizes import eta_squared, interpret_eta_squared


class TestBasicComputation:
    """Tests for basic eta-squared computations."""

    def test_two_groups_basic_eta2_float(self):
        """Test basic two-group comparison."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert isinstance(eta2, float)

    def test_two_groups_basic_eta2(self):
        """Test basic two-group comparison."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert 0 <= eta2 <= 1

    def test_three_groups_basic_eta2_float(self):
        """Test three-group ANOVA."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        group3 = np.array([5, 6, 7, 8, 9])
        # Act
        eta2 = eta_squared([group1, group2, group3])
        # Assert
        assert isinstance(eta2, float)

    def test_three_groups_basic_eta2(self):
        """Test three-group ANOVA."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        group3 = np.array([5, 6, 7, 8, 9])
        # Act
        eta2 = eta_squared([group1, group2, group3])
        # Assert
        assert 0 <= eta2 <= 1

    def test_three_groups_basic_eta2_2(self):
        """Test three-group ANOVA."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        group3 = np.array([5, 6, 7, 8, 9])
        # Act
        eta2 = eta_squared([group1, group2, group3])
        # Assert
        assert eta2 > 0.5

    def test_pandas_series_input_eta2_float(self):
        """Test that pandas Series work as input."""
        # Arrange
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([3, 4, 5, 6, 7])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert isinstance(eta2, float)

    def test_pandas_series_input_isnan_eta2(self):
        """Test that pandas Series work as input."""
        # Arrange
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([3, 4, 5, 6, 7])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert not np.isnan(eta2)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_effect_identical_groups(self):
        """Test that identical groups give η² = 0."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([1, 2, 3, 4, 5])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert abs(eta2) < 0.01  # Should be very close to 0

    def test_perfect_separation_eta2(self):
        """Test that perfect separation gives η² close to 1."""
        # Arrange
        group1 = np.array([1, 1, 1, 1, 1])
        group2 = np.array([10, 10, 10, 10, 10])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert eta2 > 0.95  # Should be very close to 1

    def test_nan_handling_eta2_float(self):
        """Test that NaN values are properly removed."""
        # Arrange
        group1 = np.array([1, 2, np.nan, 4, 5])
        group2 = np.array([3, np.nan, 5, 6, 7])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert isinstance(eta2, float)

    def test_nan_handling_isnan_eta2(self):
        """Test that NaN values are properly removed."""
        # Arrange
        group1 = np.array([1, 2, np.nan, 4, 5])
        group2 = np.array([3, np.nan, 5, 6, 7])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert not np.isnan(eta2)

    def test_all_same_values(self):
        """Test when all values across groups are identical."""
        # Arrange
        group1 = np.array([5, 5, 5])
        group2 = np.array([5, 5, 5])
        group3 = np.array([5, 5, 5])
        # Act
        eta2 = eta_squared([group1, group2, group3])
        # Assert
        assert eta2 == 0.0

    def test_single_observation_per_group(self):
        """Test with single observation per group."""
        # Arrange
        group1 = np.array([1])
        group2 = np.array([5])
        group3 = np.array([10])
        # Act
        eta2 = eta_squared([group1, group2, group3])
        # Assert
        assert 0 <= eta2 <= 1


class TestKnownValues:
    """Tests with manually calculated known values."""

    def test_known_value_simple(self):
        """Test with simple manually calculated example."""
        # Arrange
        group1 = np.array([0, 0])
        group2 = np.array([5, 5])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert abs(eta2 - 1.0) < 0.01

    def test_known_value_medium_effect(self):
        """Test with known medium effect."""
        # Arrange
        np.random.seed(42)
        n = 50
        group1 = np.random.normal(0, 1, n)
        group2 = np.random.normal(0.5, 1, n)
        group3 = np.random.normal(1.0, 1, n)
        # Act
        eta2 = eta_squared([group1, group2, group3])
        # Assert
        assert 0.1 < eta2 < 0.5


class TestMathematicalProperties:
    """Tests for mathematical properties of eta-squared."""

    def test_range_constraint_n_groups_groups_eta2_randint(self):
        """Test that η² is always between 0 and 1."""
        # Arrange
        # Act
        np.random.seed(42)
        # Assert
        for _ in range(10):
            n_groups = np.random.randint(2, 6)
            groups = [np.random.normal(i, 1, 20) for i in range(n_groups)]
            eta2 = eta_squared(groups)

            assert 0 <= eta2 <= 1

    def test_increases_with_group_separation(self):
        """Test that η² increases as groups become more separated."""
        # Arrange
        np.random.seed(42)
        base = np.random.normal(0, 1, 30)
        eta2_values = []
        # Act
        for separation in [0.0, 0.5, 1.0, 2.0]:
            g1 = np.random.normal(0, 1, 30)
            g2 = np.random.normal(separation, 1, 30)
            eta2_values.append(eta_squared([g1, g2]))
        # Assert
        assert eta2_values[-1] > eta2_values[0]

    def test_invariant_to_linear_transformation(self):
        """Test that η² is invariant to linear transformations."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        group3 = np.array([5, 6, 7, 8, 9])
        eta2_original = eta_squared([group1, group2, group3])
        scale = 10
        shift = 100
        # Act
        eta2_transformed = eta_squared(
            [group1 * scale + shift, group2 * scale + shift, group3 * scale + shift]
        )
        # Assert
        assert abs(eta2_original - eta2_transformed) < 0.001

    def test_more_groups_can_increase_eta2(self):
        """Test that adding well-separated groups increases η²."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 30)
        group2 = np.random.normal(2, 1, 30)
        eta2_two = eta_squared([group1, group2])
        group3 = np.random.normal(4, 1, 30)
        # Act
        eta2_three = eta_squared([group1, group2, group3])
        # Assert
        assert eta2_three > eta2_two * 0.8  # Allow some variation


class TestInterpretation:
    """Tests for effect size interpretation."""

    def test_interpret_negligible_interpret_eta_squared(self):
        """Test negligible effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.005) == "negligible"

    def test_interpret_negligible_interpret_eta_squared_2(self):
        """Test negligible effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.009) == "negligible"

    def test_interpret_small_interpret_eta_squared(self):
        """Test small effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.03) == "small"

    def test_interpret_small_interpret_eta_squared_2(self):
        """Test small effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.05) == "small"

    def test_interpret_medium_interpret_eta_squared(self):
        """Test medium effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.10) == "medium"

    def test_interpret_medium_interpret_eta_squared_2(self):
        """Test medium effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.12) == "medium"

    def test_interpret_large_interpret_eta_squared(self):
        """Test large effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.20) == "large"

    def test_interpret_large_interpret_eta_squared_2(self):
        """Test large effect interpretation."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.50) == "large"

    def test_interpret_boundaries_small_interpret_eta_squared(self):
        """Test interpretation at boundaries."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.01) == "small"

    def test_interpret_boundaries_medium_interpret_eta_squared(self):
        """Test interpretation at boundaries."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.06) == "medium"

    def test_interpret_boundaries_large_interpret_eta_squared(self):
        """Test interpretation at boundaries."""
        # Arrange
        # Act
        # Assert
        assert interpret_eta_squared(0.14) == "large"


class TestComparison:
    """Tests comparing eta-squared across scenarios."""

    def test_within_vs_between_variance(self):
        """Test effect of within-group vs between-group variance."""
        # Arrange
        np.random.seed(42)
        g1_high_var = np.random.normal(0, 2, 30)
        g2_high_var = np.random.normal(1, 2, 30)
        eta2_high_var = eta_squared([g1_high_var, g2_high_var])
        g1_low_var = np.random.normal(0, 0.5, 30)
        g2_low_var = np.random.normal(1, 0.5, 30)
        # Act
        eta2_low_var = eta_squared([g1_low_var, g2_low_var])
        # Assert
        assert eta2_low_var > eta2_high_var

    def test_balanced_vs_unbalanced_groups_eta2(self):
        """Test that η² works with unbalanced group sizes."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert 0 <= eta2 <= 1

    def test_balanced_vs_unbalanced_groups_eta2_2(self):
        """Test that η² works with unbalanced group sizes."""
        # Arrange
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        # Act
        eta2 = eta_squared([group1, group2])
        # Assert
        assert eta2 > 0.5  # Should show strong effect


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_eta_squared.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 21:05:00 (ywatanabe)"
# # File: ./src/scitex/stats/effect_sizes/_eta_squared.py
# # ----------------------------------------
# from __future__ import annotations
# import os
#
# __FILE__ = "./src/scitex/stats/effect_sizes/_eta_squared.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# """
# Functionalities:
#   - Compute eta-squared (η²) effect size for ANOVA
#   - Measure proportion of variance explained by group membership
#   - Provide interpretation guidelines
#   - Support multiple groups
#
# Dependencies:
#   - packages: numpy, pandas
#
# IO:
#   - input: List of sample arrays (one per group)
#   - output: Effect size value (float, ranges from 0 to 1)
# """
#
# """Imports"""
# import argparse
# from typing import List, Union
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
# def eta_squared(groups: List[Union[np.ndarray, pd.Series]], ddof: int = 1) -> float:
#     """
#     Compute eta-squared (η²) effect size for ANOVA.
#
#     Parameters
#     ----------
#     groups : list of arrays or Series
#         List of samples, one per group
#     ddof : int, default 1
#         Degrees of freedom correction for variance
#
#     Returns
#     -------
#     float
#         Eta-squared value (ranges from 0 to 1)
#
#     Notes
#     -----
#     Eta-squared (η²) measures the proportion of total variance explained
#     by group membership in ANOVA designs.
#
#     .. math::
#         \\eta^2 = \\frac{SS_{between}}{SS_{total}}
#
#     Where:
#     - SS_between: Sum of squares between groups
#     - SS_total: Total sum of squares
#
#     Interpretation (Cohen, 1988):
#     - η² < 0.01:  negligible
#     - η² < 0.06:  small
#     - η² < 0.14:  medium
#     - η² ≥ 0.14:  large
#
#     **Variants:**
#     - η²: Biased, overestimates population effect
#     - ω² (omega-squared): Less biased estimate
#     - partial η²: Used in factorial designs
#
#     Relationship to F-statistic:
#
#     .. math::
#         \\eta^2 = \\frac{F \\cdot df_{between}}{F \\cdot df_{between} + df_{within}}
#
#     References
#     ----------
#     .. [1] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
#            Sciences (2nd ed.). Routledge.
#     .. [2] Richardson, J. T. E. (2011). "Eta squared and partial eta squared
#            as measures of effect size in educational research". Educational
#            Research Review, 6(2), 135-147.
#
#     Examples
#     --------
#     >>> group1 = np.array([1, 2, 3, 4, 5])
#     >>> group2 = np.array([3, 4, 5, 6, 7])
#     >>> group3 = np.array([5, 6, 7, 8, 9])
#     >>> eta_squared([group1, group2, group3])
#     0.857...
#
#     >>> # No effect
#     >>> group1 = np.array([1, 2, 3, 4, 5])
#     >>> group2 = np.array([1, 2, 3, 4, 5])
#     >>> eta_squared([group1, group2])
#     0.0
#     """
#     # Convert all groups to numpy arrays and remove NaN
#     groups = [np.asarray(g) for g in groups]
#     groups = [g[~np.isnan(g)] for g in groups]
#
#     # Compute grand mean
#     all_data = np.concatenate(groups)
#     grand_mean = np.mean(all_data)
#
#     # Compute total sum of squares
#     ss_total = np.sum((all_data - grand_mean) ** 2)
#
#     # Compute between-group sum of squares
#     ss_between = 0
#     for group in groups:
#         group_mean = np.mean(group)
#         n_group = len(group)
#         ss_between += n_group * (group_mean - grand_mean) ** 2
#
#     # Compute eta-squared
#     if ss_total == 0:
#         return 0.0
#
#     eta2 = ss_between / ss_total
#
#     return float(eta2)
#
#
# def interpret_eta_squared(eta2: float) -> str:
#     """
#     Interpret eta-squared effect size.
#
#     Parameters
#     ----------
#     eta2 : float
#         Eta-squared value
#
#     Returns
#     -------
#     str
#         Interpretation string
#
#     Examples
#     --------
#     >>> interpret_eta_squared(0.005)
#     'negligible'
#     >>> interpret_eta_squared(0.03)
#     'small'
#     >>> interpret_eta_squared(0.10)
#     'medium'
#     >>> interpret_eta_squared(0.20)
#     'large'
#     """
#     if eta2 < 0.01:
#         return "negligible"
#     elif eta2 < 0.06:
#         return "small"
#     elif eta2 < 0.14:
#         return "medium"
#     else:
#         return "large"
#
#
# """Main function"""
#
#
# def main(args):
#     """Demonstrate eta-squared computation."""
#     logger.info("Demonstrating eta-squared effect size for ANOVA")
#
#     # Set random seed
#     np.random.seed(42)
#
#     # Example 1: Three groups with clear differences
#     logger.info("\n=== Example 1: Three groups with differences ===")
#
#     group1 = np.random.normal(0, 1, 30)
#     group2 = np.random.normal(0.5, 1, 30)
#     group3 = np.random.normal(1.0, 1, 30)
#
#     eta2 = eta_squared([group1, group2, group3])
#     interpretation = interpret_eta_squared(eta2)
#
#     logger.info(f"Eta-squared = {eta2:.3f} ({interpretation})")
#     logger.info(f"{eta2:.1%} of variance explained by group membership")
#
#     # Example 2: Different effect sizes
#     logger.info("\n=== Example 2: Different effect sizes ===")
#
#     control = np.random.normal(0, 1, 40)
#
#     for shift in [0.0, 0.3, 0.6, 1.0]:
#         treatment1 = np.random.normal(shift, 1, 40)
#         treatment2 = np.random.normal(shift * 1.5, 1, 40)
#
#         eta2 = eta_squared([control, treatment1, treatment2])
#         interpretation = interpret_eta_squared(eta2)
#
#         logger.info(f"Shift = {shift:.1f}: η² = {eta2:.3f} ({interpretation})")
#
#     # Example 3: Many groups
#     logger.info("\n=== Example 3: Five groups ===")
#
#     groups = [np.random.normal(i * 0.3, 1, 25) for i in range(5)]
#     eta2_many = eta_squared(groups)
#
#     logger.info(
#         f"Five groups: η² = {eta2_many:.3f} ({interpret_eta_squared(eta2_many)})"
#     )
#
#     # Visualization
#     logger.info("\n=== Creating visualization ===")
#
#     fig, axes = stx.plt.subplots(1, 2, figsize=(12, 5))
#
#     # Plot 1: Group distributions
#     ax = axes[0]
#     positions = [0, 1, 2]
#     data_viz = [group1, group2, group3]
#
#     bp = ax.boxplot(data_viz, positions=positions, widths=0.6, patch_artist=True)
#     for patch in bp["boxes"]:
#         patch.set_facecolor("lightblue")
#
#     ax.set_xlabel("Group")
#     ax.set_ylabel("Value")
#     ax.set_title(f"Three Groups (η² = {eta2:.2f})")
#     ax.set_xticklabels(["Group 1", "Group 2", "Group 3"])
#     ax.grid(True, alpha=0.3, axis="y")
#
#     # Plot 2: Effect size vs number of groups
#     ax = axes[1]
#     n_groups_list = [2, 3, 4, 5, 6]
#     eta2_values = []
#
#     for n_groups in n_groups_list:
#         groups_test = [np.random.normal(i * 0.4, 1, 30) for i in range(n_groups)]
#         eta2_values.append(eta_squared(groups_test))
#
#     ax.plot(n_groups_list, eta2_values, "o-", linewidth=2, markersize=8)
#     ax.set_xlabel("Number of Groups")
#     ax.set_ylabel("Eta-squared")
#     ax.set_title("Effect Size vs Number of Groups")
#     ax.grid(True, alpha=0.3)
#
#     stx.plt.tight_layout()
#     stx.io.save(fig, "./eta_squared_demo.jpg")
#     logger.info("Visualization saved")
#
#     return 0
#
#
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate eta-squared effect size calculation"
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_eta_squared.py
# --------------------------------------------------------------------------------
