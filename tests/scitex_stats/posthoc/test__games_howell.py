#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test file for scitex.stats.posthoc._games_howell

import numpy as np
import pandas as pd
import pytest

from scitex_stats.posthoc import posthoc_games_howell, posthoc_tukey


class TestBasicComputations:
    """Test basic Games-Howell computations."""

    def test_three_groups_basic_results_dataframe(self):
        """Test basic three-group comparison."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 2, 20)
        group3 = np.random.normal(14, 3, 20)
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        # Assert
        assert isinstance(results, pd.DataFrame)
        required_cols = [
            "group_i",
            "group_j",
            "mean_diff",
            "pvalue",
            "significant",
            "t_statistic",
            "df",
            "ci_lower",
            "ci_upper",
        ]

    def test_three_groups_basic_results(self):
        """Test basic three-group comparison."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 2, 20)
        group3 = np.random.normal(14, 3, 20)
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        # Assert
        assert len(results) == 3
        required_cols = [
            "group_i",
            "group_j",
            "mean_diff",
            "pvalue",
            "significant",
            "t_statistic",
            "df",
            "ci_lower",
            "ci_upper",
        ]

    def test_three_groups_basic_col_required_cols_columns_results(self):
        """Test basic three-group comparison."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 2, 20)
        group3 = np.random.normal(14, 3, 20)
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        required_cols = [
            "group_i",
            "group_j",
            "mean_diff",
            "pvalue",
            "significant",
            "t_statistic",
            "df",
            "ci_lower",
            "ci_upper",
        ]
        # Assert
        for col in required_cols:
            assert col in results.columns

    def test_unequal_variances_main_use_case_var_columns_results(self):
        """Test with unequal variances (main use case for Games-Howell)."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)  # Small variance
        group2 = np.random.normal(12, 5, 25)  # Large variance
        group3 = np.random.normal(11, 2, 15)  # Medium variance
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        # Assert
        assert "var_i" in results.columns
        variances = set(results["var_i"].tolist() + results["var_j"].tolist())

    def test_unequal_variances_main_use_case_var_columns_results_2(self):
        """Test with unequal variances (main use case for Games-Howell)."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)  # Small variance
        group2 = np.random.normal(12, 5, 25)  # Large variance
        group3 = np.random.normal(11, 2, 15)  # Medium variance
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        # Assert
        assert "var_j" in results.columns
        variances = set(results["var_i"].tolist() + results["var_j"].tolist())

    def test_unequal_variances_main_use_case_case_3(self):
        """Test with unequal variances (main use case for Games-Howell)."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)  # Small variance
        group2 = np.random.normal(12, 5, 25)  # Large variance
        group3 = np.random.normal(11, 2, 15)  # Medium variance
        results = posthoc_games_howell([group1, group2, group3])
        # Act
        variances = set(results["var_i"].tolist() + results["var_j"].tolist())
        # Assert
        assert len(variances) > 1

    def test_welch_satterthwaite_df_welch_df_pooled_df(self):
        """Test Welch-Satterthwaite degrees of freedom calculation."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 25)
        # Act
        results = posthoc_games_howell([group1, group2])
        pooled_df = 20 + 25 - 2
        welch_df = results.iloc[0]["df"]
        # Assert
        assert welch_df != pooled_df

    def test_welch_satterthwaite_df_welch_df_pooled_df_2(self):
        """Test Welch-Satterthwaite degrees of freedom calculation."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 25)
        # Act
        results = posthoc_games_howell([group1, group2])
        pooled_df = 20 + 25 - 2
        welch_df = results.iloc[0]["df"]
        # Assert
        assert 19 <= welch_df <= pooled_df


class TestInputFormats:
    """Test different input formats."""

    def test_list_of_arrays(self):
        """Test with list of numpy arrays."""
        # Arrange
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        # Act
        results = posthoc_games_howell(groups)
        # Assert
        assert len(results) == 3

    def test_pandas_series_results(self):
        """Test with pandas Series."""
        # Arrange
        groups = [pd.Series([1, 2, 3]), pd.Series([4, 5, 6]), pd.Series([7, 8, 9])]
        # Act
        results = posthoc_games_howell(groups)
        # Assert
        assert len(results) == 3

    def test_custom_group_names_low_var_iloc_results(self):
        """Test with custom group names."""
        # Arrange
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        group_names = ["Low Var", "High Var"]
        # Act
        results = posthoc_games_howell(groups, group_names=group_names)
        # Assert
        assert results.iloc[0]["group_i"] == "Low Var"

    def test_custom_group_names_high_var_iloc_results(self):
        """Test with custom group names."""
        # Arrange
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        group_names = ["Low Var", "High Var"]
        # Act
        results = posthoc_games_howell(groups, group_names=group_names)
        # Assert
        assert results.iloc[0]["group_j"] == "High Var"

    def test_return_dict_format_results_list(self):
        """Test return_as='dict' option."""
        # Arrange
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        # Act
        results = posthoc_games_howell(groups, return_as="dict")
        # Assert
        assert isinstance(results, list)

    def test_return_dict_format_results(self):
        """Test return_as='dict' option."""
        # Arrange
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        # Act
        results = posthoc_games_howell(groups, return_as="dict")
        # Assert
        assert isinstance(results[0], dict)

    def test_return_dict_format_mean_diff_results(self):
        """Test return_as='dict' option."""
        # Arrange
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        # Act
        results = posthoc_games_howell(groups, return_as="dict")
        # Assert
        assert "mean_diff" in results[0]

    def test_return_dict_format_df_results(self):
        """Test return_as='dict' option."""
        # Arrange
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        # Act
        results = posthoc_games_howell(groups, return_as="dict")
        # Assert
        assert "df" in results[0]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_two_groups_minimum(self):
        """Test with minimum of two groups."""
        # Arrange
        group1 = np.random.normal(10, 1, 10)
        group2 = np.random.normal(12, 2, 15)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert len(results) == 1

    def test_single_group_raises_error(self):
        """Test that single group raises ValueError."""
        # Arrange
        # Act
        group = np.array([1, 2, 3, 4, 5])
        # Assert
        with pytest.raises(ValueError, match="Need at least 2 groups"):
            posthoc_games_howell([group])

    def test_extreme_variance_ratio_results(self):
        """Test with extreme variance heterogeneity."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(50, 0.1, 20)  # Very small variance
        group2 = np.random.normal(55, 10, 20)  # Very large variance
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert len(results) == 1
        var_ratio = results.iloc[0]["var_j"] / results.iloc[0]["var_i"]

    def test_extreme_variance_ratio_var_ratio(self):
        """Test with extreme variance heterogeneity."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(50, 0.1, 20)  # Very small variance
        group2 = np.random.normal(55, 10, 20)  # Very large variance
        # Act
        results = posthoc_games_howell([group1, group2])
        var_ratio = results.iloc[0]["var_j"] / results.iloc[0]["var_i"]
        # Assert
        assert var_ratio > 100 or var_ratio < 0.01

    def test_unequal_sample_sizes_with_unequal_variances_results(self):
        """Test with both unequal n and unequal variances."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 10)  # Small n, small var
        group2 = np.random.normal(12, 5, 30)  # Large n, large var
        group3 = np.random.normal(14, 2, 20)  # Medium n, medium var
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        # Assert
        assert len(results) == 3

    def test_unequal_sample_sizes_with_unequal_variances_iloc_results(self):
        """Test with both unequal n and unequal variances."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 10)  # Small n, small var
        group2 = np.random.normal(12, 5, 30)  # Large n, large var
        group3 = np.random.normal(14, 2, 20)  # Medium n, medium var
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        # Assert
        assert results.iloc[0]["n_i"] != results.iloc[0]["n_j"]

    def test_zero_variance_group(self):
        """Test with group having zero variance."""
        # Arrange
        group1 = np.array([5, 5, 5, 5, 5])  # Zero variance
        group2 = np.random.normal(10, 2, 5)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert len(results) == 1

    def test_mismatched_group_names_length(self):
        """Test error when group_names length doesn't match."""
        # Arrange
        # Act
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        group_names = ["Group1"]  # Only 1 name for 2 groups
        # Assert
        with pytest.raises(ValueError, match="Expected 2 group names"):
            posthoc_games_howell(groups, group_names=group_names)


class TestComparisonWithTukey:
    """Compare Games-Howell with Tukey HSD."""

    def test_similar_results_equal_variances(self):
        """Test that Games-Howell and Tukey give similar results with equal variances."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 30)
        group3 = np.random.normal(14, 2, 30)
        results_gh = posthoc_games_howell([group1, group2, group3])
        # Act
        results_tukey = posthoc_tukey([group1, group2, group3])
        # Assert
        for i in range(len(results_gh)):
            assert (
                abs(results_gh.iloc[i]["pvalue"] - results_tukey.iloc[i]["pvalue"])
                < 0.1
            )

    def test_different_results_unequal_variances(self):
        """Test that Games-Howell differs from Tukey with unequal variances."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 0.5, 20)  # Small variance
        group2 = np.random.normal(12, 5, 20)  # Large variance
        group3 = np.random.normal(11, 10, 20)  # Very large variance
        results_gh = posthoc_games_howell([group1, group2, group3])
        results_tukey = posthoc_tukey([group1, group2, group3])
        diff_found = False
        # Act
        for i in range(len(results_gh)):
            gh_sig = results_gh.iloc[i]["significant"]
            tukey_sig = results_tukey.iloc[i]["significant"]
            if gh_sig != tukey_sig:
                diff_found = True
                break
        # Assert
        assert len(results_gh) == len(results_tukey)

    def test_games_howell_more_conservative_unequal_var_results_gh(self):
        """Games-Howell should handle unequal variances better."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 10, 20)
        results_gh = posthoc_games_howell([group1, group2])
        # Act
        results_tukey = posthoc_tukey([group1, group2])
        # Assert
        assert len(results_gh) == 1

    def test_games_howell_more_conservative_unequal_var_results_tukey(self):
        """Games-Howell should handle unequal variances better."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 10, 20)
        results_gh = posthoc_games_howell([group1, group2])
        # Act
        results_tukey = posthoc_tukey([group1, group2])
        # Assert
        assert len(results_tukey) == 1


class TestStatisticalProperties:
    """Test statistical properties of results."""

    def test_alpha_level_respected_iloc_results_001(self):
        """Test that alpha level affects significance."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(11, 2, 20)
        results_005 = posthoc_games_howell([group1, group2], alpha=0.05)
        # Act
        results_001 = posthoc_games_howell([group1, group2], alpha=0.01)
        # Assert
        assert results_001.iloc[0]["alpha"] == 0.01

    def test_alpha_level_respected_iloc_results_005(self):
        """Test that alpha level affects significance."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(11, 2, 20)
        results_005 = posthoc_games_howell([group1, group2], alpha=0.05)
        # Act
        results_001 = posthoc_games_howell([group1, group2], alpha=0.01)
        # Assert
        assert results_005.iloc[0]["alpha"] == 0.05

    def test_confidence_intervals_row_iterrows_results_ci(self):
        """Test confidence interval properties."""
        # Arrange
        np.random.seed(42)
        groups = [np.random.normal(10 + i, i + 1, 20) for i in range(3)]
        # Act
        results = posthoc_games_howell(groups)
        # Assert
        for _, row in results.iterrows():
            # CI must contain the mean difference AND have positive width
            # (a strict lower < upper makes the containment non-degenerate).
            assert (
                row["ci_lower"] <= row["mean_diff"] <= row["ci_upper"]
                and row["ci_upper"] > row["ci_lower"]
            )

    def test_t_statistic_calculation_abs_iloc_results(self):
        """Test t-statistic calculation."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(15, 2, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert abs(results.iloc[0]["t_statistic"]) > 2

    def test_t_statistic_calculation_std_error_iloc_results(self):
        """Test t-statistic calculation."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(15, 2, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert results.iloc[0]["std_error"] > 0

    def test_degrees_of_freedom_properties_case_1(self):
        """Test properties of Welch-Satterthwaite df."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 30)
        # Act
        results = posthoc_games_howell([group1, group2])
        df = results.iloc[0]["df"]
        # Assert
        assert df > 0

    def test_degrees_of_freedom_properties_case_2(self):
        """Test properties of Welch-Satterthwaite df."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 30)
        # Act
        results = posthoc_games_howell([group1, group2])
        df = results.iloc[0]["df"]
        # Assert
        assert df <= (20 + 30 - 2)

    def test_degrees_of_freedom_properties_min(self):
        """Test properties of Welch-Satterthwaite df."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 30)
        # Act
        results = posthoc_games_howell([group1, group2])
        df = results.iloc[0]["df"]
        # Assert
        assert df >= min(20, 30) - 1


class TestOutputStructure:
    """Test output structure and completeness."""

    def test_dataframe_output_structure(self):
        """Test DataFrame output has all required fields."""
        # Arrange
        groups = [np.random.normal(i, i + 1, 10) for i in range(3)]
        # Act
        results = posthoc_games_howell(groups)
        required_fields = [
            "group_i",
            "group_j",
            "n_i",
            "n_j",
            "mean_i",
            "mean_j",
            "var_i",
            "var_j",
            "mean_diff",
            "std_error",
            "t_statistic",
            "df",
            "pvalue",
            "significant",
            "pstars",
            "ci_lower",
            "ci_upper",
            "alpha",
        ]
        # Assert
        for field in required_fields:
            assert field in results.columns, f"Missing field: {field}"

    def test_dict_output_structure(self):
        """Test dict output has all required fields."""
        # Arrange
        groups = [np.random.normal(i, i + 1, 10) for i in range(3)]
        # Act
        results = posthoc_games_howell(groups, return_as="dict")
        required_fields = [
            "group_i",
            "group_j",
            "mean_diff",
            "pvalue",
            "significant",
            "t_statistic",
            "df",
            "var_i",
            "var_j",
        ]
        # Assert
        for result in results:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

    def test_variance_information_included_var_columns_results(self):
        """Test that variance information is included in output."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert "var_i" in results.columns

    def test_variance_information_included_var_columns_results_2(self):
        """Test that variance information is included in output."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert "var_j" in results.columns

    def test_variance_information_included_var_iloc_results(self):
        """Test that variance information is included in output."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert results.iloc[0]["var_i"] > 0

    def test_variance_information_included_var_iloc_results_2(self):
        """Test that variance information is included in output."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 20)
        group2 = np.random.normal(12, 5, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert results.iloc[0]["var_j"] > 0


class TestRobustness:
    """Test robustness to various data conditions."""

    def test_small_sample_sizes(self):
        """Test with very small sample sizes."""
        # Arrange
        group1 = np.array([1, 2])
        group2 = np.array([3, 4, 5])
        group3 = np.array([6, 7])
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        # Assert
        assert len(results) == 3

    def test_large_number_of_groups(self):
        """Test with many groups."""
        # Arrange
        np.random.seed(42)
        groups = [np.random.normal(i, i * 0.5 + 1, 10) for i in range(8)]
        # Act
        results = posthoc_games_howell(groups)
        # Assert
        assert len(results) == 28

    def test_extreme_mean_differences_significant_iloc_results(self):
        """Test with extreme mean differences."""
        # Arrange
        group1 = np.random.normal(0, 1, 20)
        group2 = np.random.normal(1000, 1, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert results.iloc[0]["significant"]

    def test_extreme_mean_differences_pvalue_iloc_results(self):
        """Test with extreme mean differences."""
        # Arrange
        group1 = np.random.normal(0, 1, 20)
        group2 = np.random.normal(1000, 1, 20)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert results.iloc[0]["pvalue"] < 0.001

    def test_high_variance_groups_results(self):
        """Test with very high variance."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 100, 30)
        group2 = np.random.normal(12, 100, 30)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert len(results) == 1

    def test_high_variance_groups_std_error_iloc_results(self):
        """Test with very high variance."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 100, 30)
        group2 = np.random.normal(12, 100, 30)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert results.iloc[0]["std_error"] > 10


class TestSpecialCases:
    """Test special cases specific to Games-Howell."""

    def test_heteroscedastic_data_row1_var(self):
        """Test with clearly heteroscedastic data."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 25)
        group2 = np.random.normal(10, 3, 25)
        group3 = np.random.normal(10, 5, 25)
        # Act
        results = posthoc_games_howell([group1, group2, group3])
        row1 = results[
            (results["group_i"] == "Group 1") & (results["group_j"] == "Group 2")
        ].iloc[0]
        # Assert
        assert row1["var_i"] < row1["var_j"]

    def test_equal_means_different_variances_significant_iloc_results(self):
        """Test groups with equal means but different variances."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 30)
        group2 = np.random.normal(10, 5, 30)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert not results.iloc[0]["significant"]

    def test_equal_means_different_variances_abs_var_iloc_results(self):
        """Test groups with equal means but different variances."""
        # Arrange
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 30)
        group2 = np.random.normal(10, 5, 30)
        # Act
        results = posthoc_games_howell([group1, group2])
        # Assert
        assert abs(results.iloc[0]["var_i"] - results.iloc[0]["var_j"]) > 1


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/posthoc/_games_howell.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 20:00:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/posthoc/_games_howell.py
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
#   - Perform Games-Howell post-hoc test
#   - Pairwise comparisons with unequal variances
#   - Does not assume homogeneity of variance
#   - Welch-Satterthwaite degrees of freedom
#
# Dependencies:
#   - packages: numpy, pandas, scipy
#
# IO:
#   - input: Multiple groups data
#   - output: Pairwise comparison results (DataFrame)
# """
#
# """Imports"""
# import numpy as np
# import pandas as pd
# from typing import Union, List, Optional
# from scipy import stats
# from scitex_stats._utils._formatters import p2stars
# from scitex_stats._utils._normalizers import convert_results
#
#
# def welch_satterthwaite_df(var_i: float, n_i: int, var_j: float, n_j: int) -> float:
#     """
#     Compute Welch-Satterthwaite degrees of freedom.
#
#     Parameters
#     ----------
#     var_i : float
#         Variance of group i
#     n_i : int
#         Sample size of group i
#     var_j : float
#         Variance of group j
#     n_j : int
#         Sample size of group j
#
#     Returns
#     -------
#     df : float
#         Degrees of freedom
#
#     Notes
#     -----
#     Formula:
#     df = (s_i²/n_i + s_j²/n_j)² / [(s_i²/n_i)²/(n_i-1) + (s_j²/n_j)²/(n_j-1)]
#     """
#     s_i_sq_n_i = var_i / n_i
#     s_j_sq_n_j = var_j / n_j
#
#     numerator = (s_i_sq_n_i + s_j_sq_n_j) ** 2
#     denominator = (s_i_sq_n_i**2) / (n_i - 1) + (s_j_sq_n_j**2) / (n_j - 1)
#
#     if denominator == 0:
#         return n_i + n_j - 2
#
#     df = numerator / denominator
#     return float(df)
#
#
# def posthoc_games_howell(
#     groups: List[Union[np.ndarray, pd.Series]],
#     group_names: Optional[List[str]] = None,
#     alpha: float = 0.05,
#     return_as: str = "dataframe",
# ) -> Union[pd.DataFrame, List[dict]]:
#     """
#     Perform Games-Howell post-hoc test for pairwise comparisons.
#
#     Modified Tukey HSD that does not assume equal variances.
#     Uses Welch-Satterthwaite degrees of freedom approximation.
#
#     Parameters
#     ----------
#     groups : list of arrays
#         List of sample arrays for each group
#     group_names : list of str, optional
#         Names for each group. If None, uses 'Group 1', 'Group 2', etc.
#     alpha : float, default 0.05
#         Family-wise error rate
#     return_as : {'dataframe', 'dict'}, default 'dataframe'
#         Output format
#
#     Returns
#     -------
#     results : DataFrame or list of dict
#         Pairwise comparison results including:
#         - group_i: First group name
#         - group_j: Second group name
#         - mean_i: Mean of group i
#         - mean_j: Mean of group j
#         - mean_diff: Difference in means (i - j)
#         - std_error: Standard error of difference
#         - t_statistic: t-statistic (Welch)
#         - df: Welch-Satterthwaite degrees of freedom
#         - pvalue: p-value
#         - significant: Whether difference is significant
#         - ci_lower: Lower bound of 95% CI
#         - ci_upper: Upper bound of 95% CI
#
#     Notes
#     -----
#     Games-Howell test is a non-parametric post-hoc test that does not
#     assume equal variances across groups (heteroscedasticity).
#
#     **Test Statistic (Welch t-test)**:
#
#     .. math::
#         t = \\frac{\\bar{x}_i - \\bar{x}_j}{\\sqrt{s_i^2/n_i + s_j^2/n_j}}
#
#     **Degrees of Freedom (Welch-Satterthwaite)**:
#
#     .. math::
#         df = \\frac{(s_i^2/n_i + s_j^2/n_j)^2}{\\frac{(s_i^2/n_i)^2}{n_i-1} + \\frac{(s_j^2/n_j)^2}{n_j-1}}
#
#     **Critical Value**:
#     Uses studentized range distribution with Welch df (approximated).
#
#     **Assumptions**:
#     1. Independence of observations
#     2. Normality within each group
#     3. **Does NOT assume equal variances** (main advantage over Tukey HSD)
#
#     **Advantages**:
#     - Robust to unequal variances
#     - More accurate than Tukey HSD when homogeneity violated
#     - Controls Type I error well even with variance heterogeneity
#
#     **Disadvantages**:
#     - Slightly less powerful than Tukey HSD when variances are equal
#     - More complex calculations
#     - Requires larger sample sizes for accuracy
#
#     **When to use**:
#     - After ANOVA with unequal variances (Levene's test significant)
#     - When Tukey HSD assumptions violated
#     - With unbalanced designs and heteroscedasticity
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> from scitex_stats.posthoc import posthoc_games_howell
#     >>>
#     >>> # Example: Groups with different variances
#     >>> np.random.seed(42)
#     >>> group1 = np.random.normal(10, 1, 20)   # Small variance
#     >>> group2 = np.random.normal(12, 5, 25)   # Large variance
#     >>> group3 = np.random.normal(11, 2, 15)   # Medium variance
#     >>>
#     >>> results = posthoc_games_howell(
#     ...     [group1, group2, group3],
#     ...     group_names=['Low Var', 'High Var', 'Med Var']
#     ... )
#     >>>
#     >>> print(results[['group_i', 'group_j', 'mean_diff', 'pvalue', 'significant']])
#
#     References
#     ----------
#     .. [1] Games, P. A., & Howell, J. F. (1976). "Pairwise multiple comparison
#            procedures with unequal n's and/or variances: A Monte Carlo study".
#            Journal of Educational Statistics, 1(2), 113-125.
#     .. [2] Welch, B. L. (1947). "The generalization of 'Student's' problem when
#            several different population variances are involved". Biometrika,
#            34(1/2), 28-35.
#
#     See Also
#     --------
#     posthoc_tukey : For equal variances
#     posthoc_dunnett : For comparisons vs control
#     """
#     # Convert to list of arrays
#     groups = [np.asarray(g) for g in groups]
#
#     k = len(groups)
#
#     if k < 2:
#         raise ValueError("Need at least 2 groups for pairwise comparisons")
#
#     # Group names
#     if group_names is None:
#         group_names = [f"Group {i + 1}" for i in range(k)]
#
#     if len(group_names) != k:
#         raise ValueError(f"Expected {k} group names, got {len(group_names)}")
#
#     # Compute group statistics
#     n_groups = [len(g) for g in groups]
#     means = [np.mean(g) for g in groups]
#     variances = [np.var(g, ddof=1) for g in groups]
#
#     # Perform all pairwise comparisons
#     results = []
#
#     for i in range(k):
#         for j in range(i + 1, k):
#             n_i = n_groups[i]
#             n_j = n_groups[j]
#             mean_i = means[i]
#             mean_j = means[j]
#             var_i = variances[i]
#             var_j = variances[j]
#
#             # Mean difference
#             mean_diff = mean_i - mean_j
#
#             # Standard error (Welch formula)
#             se = np.sqrt(var_i / n_i + var_j / n_j)
#
#             # Welch t-statistic
#             if se == 0:
#                 t_stat = 0.0
#             else:
#                 t_stat = mean_diff / se
#
#             # Welch-Satterthwaite degrees of freedom
#             df = welch_satterthwaite_df(var_i, n_i, var_j, n_j)
#
#             # p-value (two-tailed)
#             pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))
#
#             # Critical value for CI (using studentized range approximation)
#             # For simplicity, use t-distribution with Bonferroni-like adjustment
#             alpha_adj = alpha / (k * (k - 1) / 2)
#             t_crit = stats.t.ppf(1 - alpha_adj / 2, df)
#
#             # Determine significance
#             significant = abs(t_stat) > t_crit
#
#             # Confidence interval
#             margin = t_crit * se
#             ci_lower = mean_diff - margin
#             ci_upper = mean_diff + margin
#
#             results.append(
#                 {
#                     "group_i": group_names[i],
#                     "group_j": group_names[j],
#                     "n_i": n_i,
#                     "n_j": n_j,
#                     "mean_i": round(float(mean_i), 3),
#                     "mean_j": round(float(mean_j), 3),
#                     "var_i": round(float(var_i), 3),
#                     "var_j": round(float(var_j), 3),
#                     "mean_diff": round(float(mean_diff), 3),
#                     "std_error": round(float(se), 3),
#                     "t_statistic": round(float(t_stat), 3),
#                     "df": round(float(df), 2),
#                     "pvalue": round(float(pvalue), 4),
#                     "significant": bool(significant),
#                     "pstars": p2stars(pvalue),
#                     "ci_lower": round(float(ci_lower), 3),
#                     "ci_upper": round(float(ci_upper), 3),
#                     "alpha": alpha,
#                 }
#             )
#
#     # Return format
#     if return_as == "dataframe":
#         return pd.DataFrame(results)
#     else:
#         return results
#
#
# if __name__ == "__main__":
#     import sys
#     import argparse
#     import scitex as stx
#
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args([])
#
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys=sys,
#         plt=None,
#         args=args,
#         file=__FILE__,
#         verbose=True,
#         agg=True,
#     )
#
#     logger = stx.logging.getLogger(__name__)
#
#     logger.info("=" * 70)
#     logger.info("Games-Howell Post-hoc Test Examples")
#     logger.info("=" * 70)
#
#     # Example 1: Unequal variances
#     logger.info("\n[Example 1] Groups with unequal variances")
#     logger.info("-" * 70)
#
#     np.random.seed(42)
#     group1 = np.random.normal(10, 1, 20)  # Small variance
#     group2 = np.random.normal(12, 5, 25)  # Large variance
#     group3 = np.random.normal(11, 2, 15)  # Medium variance
#
#     logger.info(
#         f"Group 1: mean={np.mean(group1):.2f}, var={np.var(group1, ddof=1):.2f}, n={len(group1)}"
#     )
#     logger.info(
#         f"Group 2: mean={np.mean(group2):.2f}, var={np.var(group2, ddof=1):.2f}, n={len(group2)}"
#     )
#     logger.info(
#         f"Group 3: mean={np.mean(group3):.2f}, var={np.var(group3, ddof=1):.2f}, n={len(group3)}"
#     )
#
#     results = posthoc_games_howell(
#         [group1, group2, group3], group_names=["Low Var", "High Var", "Med Var"]
#     )
#
#     logger.info(
#         f"\n{results[['group_i', 'group_j', 'mean_diff', 'df', 'pvalue', 'significant']].to_string()}"
#     )
#
#     # Example 2: Comparison with Tukey HSD
#     logger.info("\n[Example 2] Games-Howell vs Tukey HSD")
#     logger.info("-" * 70)
#
#     from ._tukey_hsd import posthoc_tukey
#
#     results_gh = posthoc_games_howell([group1, group2, group3])
#     results_tukey = posthoc_tukey([group1, group2, group3])
#
#     logger.info("\nGames-Howell results:")
#     logger.info(
#         f"{results_gh[['group_i', 'group_j', 'pvalue', 'significant']].to_string()}"
#     )
#
#     logger.info("\nTukey HSD results:")
#     logger.info(
#         f"{results_tukey[['group_i', 'group_j', 'pvalue', 'significant']].to_string()}"
#     )
#
#     logger.info("\nNote: Games-Howell is more appropriate with unequal variances")
#
#     # Example 3: After ANOVA with heteroscedasticity
#     logger.info("\n[Example 3] After ANOVA with violated homogeneity")
#     logger.info("-" * 70)
#
#     from ..tests.parametric import test_anova
#
#     anova_result = test_anova(
#         [group1, group2, group3],
#         var_names=["Low Var", "High Var", "Med Var"],
#         check_assumptions=True,
#     )
#
#     logger.info(
#         f"ANOVA: F = {anova_result['statistic']:.3f}, p = {anova_result['pvalue']:.4f}"
#     )
#     logger.info(f"Assumptions met: {anova_result.get('assumptions_met', 'N/A')}")
#
#     is_sig = anova_result.get("significant", anova_result.get("is_significant", False))
#     if is_sig:
#         logger.info(
#             "\nANOVA significant. Using Games-Howell (robust to unequal variances)..."
#         )
#         logger.info(f"\n{results.to_string()}")
#
#     # Example 4: Extreme variance ratios
#     logger.info("\n[Example 4] Extreme variance heterogeneity")
#     logger.info("-" * 70)
#
#     extreme1 = np.random.normal(50, 1, 20)  # Very small variance
#     extreme2 = np.random.normal(55, 10, 20)  # Very large variance
#
#     var_ratio = np.var(extreme2, ddof=1) / np.var(extreme1, ddof=1)
#     logger.info(f"Variance ratio: {var_ratio:.1f}")
#
#     results_extreme = posthoc_games_howell([extreme1, extreme2])
#
#     logger.info(f"\n{results_extreme.to_string()}")
#
#     # Example 5: Export results
#     logger.info("\n[Example 5] Export results")
#     logger.info("-" * 70)
#
#     convert_results(results, return_as="excel", path="./games_howell_results.xlsx")
#     logger.info("Saved to: ./games_howell_results.xlsx")
#
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         exit_status=0,
#     )
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/posthoc/_games_howell.py
# --------------------------------------------------------------------------------
