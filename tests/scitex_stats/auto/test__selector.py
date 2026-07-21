#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for test selector functions.

Tests cover:
- check_applicable() returns (bool, List[str]) with correct reasons
- get_menu_items() returns properly structured menu items
- recommend_tests() returns top-k tests sorted by priority
- Brunner-Munzel is recommended first for 2-group between-subjects
- Tests are correctly disabled with appropriate tooltips when assumptions fail
- Family filtering for recommendations and menu generation
"""

import pytest

from scitex_stats.auto._context import StatContext
from scitex_stats.auto._rules import TEST_RULES
from scitex_stats.auto._selector import (
    check_applicable,
    get_menu_items,
    recommend_effect_sizes,
    recommend_posthoc,
    recommend_tests,
)


class TestCheckApplicable:
    """Tests for check_applicable() function."""

    def test_returns_tuple_with_bool_and_list_ok(self):
        """Test that check_applicable returns (bool, list) tuple."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert isinstance(ok, bool)

    def test_returns_tuple_with_bool_and_list_reasons(self):
        """Test that check_applicable returns (bool, list) tuple."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert isinstance(reasons, list)

    def test_returns_tuple_with_bool_and_list_all_str_reasons(self):
        """Test that check_applicable returns (bool, list) tuple."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert all(isinstance(r, str) for r in reasons)

    def test_ttest_ind_applicable_with_normality_ok(self):
        """Test that t-test is applicable when normality is met."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is True

    def test_ttest_ind_applicable_with_normality_reasons(self):
        """Test that t-test is applicable when normality is met."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) == 0

    def test_ttest_ind_not_applicable_without_normality_ok(self):
        """Test that t-test is not applicable when normality fails."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_ttest_ind_not_applicable_without_normality_reasons(self):
        """Test that t-test is not applicable when normality fails."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_ttest_ind_not_applicable_without_normality_any_reasons_lower(self):
        """Test that t-test is not applicable when normality fails."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("normality" in r.lower() for r in reasons)

    def test_brunner_munzel_applicable_without_assumptions_ok(self):
        """Test that Brunner-Munzel is applicable without assumptions."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
            variance_homogeneity_ok=False,
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is True

    def test_brunner_munzel_applicable_without_assumptions_reasons(self):
        """Test that Brunner-Munzel is applicable without assumptions."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
            variance_homogeneity_ok=False,
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) == 0

    def test_not_applicable_wrong_group_count_too_few_ok(self):
        """Test rejection when too few groups."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_wrong_group_count_too_few_reasons(self):
        """Test rejection when too few groups."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_wrong_group_count_too_few_any_reasons_at_least(self):
        """Test rejection when too few groups."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("at least" in r.lower() and "groups" in r.lower() for r in reasons)

    def test_not_applicable_wrong_group_count_too_many_ok(self):
        """Test rejection when too many groups."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_wrong_group_count_too_many_reasons(self):
        """Test rejection when too many groups."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_wrong_group_count_too_many_any_reasons_maximum_groups(self):
        """Test rejection when too many groups."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("maximum" in r.lower() and "groups" in r.lower() for r in reasons)

    def test_not_applicable_wrong_outcome_type_ok(self):
        """Test rejection for wrong outcome type."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_wrong_outcome_type_reasons(self):
        """Test rejection for wrong outcome type."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_wrong_outcome_type_any_continuous_reasons_lower(self):
        """Test rejection for wrong outcome type."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("continuous" in r.lower() for r in reasons)

    def test_not_applicable_paired_mismatch_ok(self):
        """Test rejection for paired/unpaired mismatch."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_paired_mismatch_reasons(self):
        """Test rejection for paired/unpaired mismatch."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_paired_mismatch_any_reasons_independent_lower(self):
        """Test rejection for paired/unpaired mismatch."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("paired" in r.lower() or "independent" in r.lower() for r in reasons)

    def test_not_applicable_design_mismatch_ok(self):
        """Test rejection for design mismatch."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="within",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_design_mismatch_reasons(self):
        """Test rejection for design mismatch."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="within",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_design_mismatch_any_reasons_lower(self):
        """Test rejection for design mismatch."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="within",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("design" in r.lower() for r in reasons)

    def test_not_applicable_sample_size_too_small_total_ok(self):
        """Test rejection for insufficient total sample size."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 1],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_sample_size_too_small_total_reasons(self):
        """Test rejection for insufficient total sample size."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 1],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_sample_size_too_small_total_any_reasons_lower(self):
        """Test rejection for insufficient total sample size."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 1],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("sample size" in r.lower() for r in reasons)

    def test_not_applicable_sample_size_too_small_per_group_ok(self):
        """Test rejection for insufficient per-group sample size."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 100],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_sample_size_too_small_per_group_reasons(self):
        """Test rejection for insufficient per-group sample size."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 100],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_sample_size_too_small_per_group_any_reasons_each_smallest(self):
        """Test rejection for insufficient per-group sample size."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 100],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any(
            "each group" in r.lower() or "smallest group" in r.lower() for r in reasons
        )

    def test_not_applicable_equal_variance_assumption_fails_ok(self):
        """Test rejection when equal variance assumption fails."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=False,
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_equal_variance_assumption_fails_reasons(self):
        """Test rejection when equal variance assumption fails."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=False,
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_equal_variance_assumption_fails_any_reasons_lower(self):
        """Test rejection when equal variance assumption fails."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=False,
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any(
            "equal variance" in r.lower() or "variance" in r.lower() for r in reasons
        )

    def test_not_applicable_requires_control_group_ok(self):
        """Test rejection when control group is required but missing."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            has_control_group=False,
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        rule = TEST_RULES["dunnett"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_requires_control_group_reasons(self):
        """Test rejection when control group is required but missing."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            has_control_group=False,
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        rule = TEST_RULES["dunnett"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_requires_control_group_any_reasons_lower(self):
        """Test rejection when control group is required but missing."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            has_control_group=False,
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        rule = TEST_RULES["dunnett"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("control group" in r.lower() for r in reasons)

    def test_not_applicable_too_few_factors_ok(self):
        """Test rejection for insufficient factors."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            n_factors=1,
        )
        rule = TEST_RULES["anova_twoway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_too_few_factors_reasons(self):
        """Test rejection for insufficient factors."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            n_factors=1,
        )
        rule = TEST_RULES["anova_twoway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_too_few_factors_any_factor_reasons_lower(self):
        """Test rejection for insufficient factors."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            n_factors=1,
        )
        rule = TEST_RULES["anova_twoway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("factor" in r.lower() for r in reasons)

    def test_not_applicable_too_many_factors_ok(self):
        """Test rejection for too many factors."""
        # Arrange
        ctx = StatContext(
            n_groups=8,
            sample_sizes=[10] * 8,
            outcome_type="continuous",
            design="between",
            n_factors=3,
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_not_applicable_too_many_factors_reasons(self):
        """Test rejection for too many factors."""
        # Arrange
        ctx = StatContext(
            n_groups=8,
            sample_sizes=[10] * 8,
            outcome_type="continuous",
            design="between",
            n_factors=3,
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) > 0

    def test_not_applicable_too_many_factors_any_factor_reasons_lower(self):
        """Test rejection for too many factors."""
        # Arrange
        ctx = StatContext(
            n_groups=8,
            sample_sizes=[10] * 8,
            outcome_type="continuous",
            design="between",
            n_factors=3,
        )
        rule = TEST_RULES["anova_oneway"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert any("factor" in r.lower() for r in reasons)

    def test_multiple_reasons_for_rejection_ok(self):
        """Test that multiple rejection reasons are collected."""
        # Arrange
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[5],
            outcome_type="categorical",
            design="within",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False

    def test_multiple_reasons_for_rejection_case_2(self):
        """Test that multiple rejection reasons are collected."""
        # Arrange
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[5],
            outcome_type="categorical",
            design="within",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) >= 2


class TestGetMenuItems:
    """Tests for get_menu_items() function."""

    def test_returns_list_of_dicts_items(self):
        """Test that get_menu_items returns a list of dictionaries."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        # Assert
        assert isinstance(items, list)

    def test_returns_list_of_dicts_all_item_dict_items(self):
        """Test that get_menu_items returns a list of dictionaries."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        # Assert
        assert all(isinstance(item, dict) for item in items)

    def test_menu_items_have_required_fields(self):
        """Test that menu items have all required fields."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        required_fields = ["id", "label", "family", "enabled", "tooltip", "priority"]
        # Assert
        for item in items:
            for field in required_fields:
                assert field in item, f"Item should have '{field}' field"

    def test_menu_items_enabled_for_brunner_munzel_bm_items(self):
        """Test that Brunner-Munzel is enabled for 2-group comparison."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        bm_items = [i for i in items if i["id"] == "brunner_munzel"]
        # Assert
        assert len(bm_items) == 1

    def test_menu_items_enabled_for_brunner_munzel_bm_items_2(self):
        """Test that Brunner-Munzel is enabled for 2-group comparison."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        bm_items = [i for i in items if i["id"] == "brunner_munzel"]
        # Assert
        assert bm_items[0]["enabled"] is True

    def test_menu_items_enabled_for_brunner_munzel_tooltip_bm_items(self):
        """Test that Brunner-Munzel is enabled for 2-group comparison."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        bm_items = [i for i in items if i["id"] == "brunner_munzel"]
        # Assert
        assert bm_items[0]["tooltip"] is None

    def test_menu_items_disabled_with_tooltip_ttest_items(self):
        """Test that disabled items have tooltip explaining why."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        # Act
        items = get_menu_items(ctx)
        ttest_items = [i for i in items if i["id"] == "ttest_ind"]
        # Assert
        assert len(ttest_items) == 1

    def test_menu_items_disabled_with_tooltip_enabled_ttest_items(self):
        """Test that disabled items have tooltip explaining why."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        # Act
        items = get_menu_items(ctx)
        ttest_items = [i for i in items if i["id"] == "ttest_ind"]
        # Assert
        assert ttest_items[0]["enabled"] is False

    def test_menu_items_disabled_with_tooltip_ttest_items_2(self):
        """Test that disabled items have tooltip explaining why."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        # Act
        items = get_menu_items(ctx)
        ttest_items = [i for i in items if i["id"] == "ttest_ind"]
        # Assert
        assert ttest_items[0]["tooltip"] is not None

    def test_menu_items_disabled_with_tooltip_normality_lower_ttest_items(self):
        """Test that disabled items have tooltip explaining why."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        # Act
        items = get_menu_items(ctx)
        ttest_items = [i for i in items if i["id"] == "ttest_ind"]
        # Assert
        assert "normality" in ttest_items[0]["tooltip"].lower()

    def test_menu_items_sorted_enabled_first(self):
        """Test that enabled items appear before disabled items."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        items = get_menu_items(ctx)
        enabled_indices = [i for i, item in enumerate(items) if item["enabled"]]
        # Act
        disabled_indices = [i for i, item in enumerate(items) if not item["enabled"]]
        # Assert
        if enabled_indices and disabled_indices:
            assert max(enabled_indices) < min(disabled_indices)

    def test_menu_items_sorted_by_priority(self):
        """Test that enabled items are sorted by priority (descending)."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        enabled_items = [i for i in items if i["enabled"]]
        priorities = [item["priority"] for item in enabled_items]
        # Assert
        assert priorities == sorted(priorities, reverse=True)

    def test_menu_items_include_families_filter_all_nonparametric_item_family(self):
        """Test filtering by include_families."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx, include_families=["nonparametric"])
        # Assert
        assert all(item["family"] == "nonparametric" for item in items)

    def test_menu_items_include_families_filter_any_brunner_munzel_item(self):
        """Test filtering by include_families."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx, include_families=["nonparametric"])
        # Assert
        assert any(item["id"] == "brunner_munzel" for item in items)

    def test_menu_items_include_families_filter_any_ttest_ind_item(self):
        """Test filtering by include_families."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx, include_families=["nonparametric"])
        # Assert
        assert not any(item["id"] == "ttest_ind" for item in items)

    def test_menu_items_exclude_families_filter_all_parametric_item_family(self):
        """Test filtering by exclude_families."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx, exclude_families=["parametric"])
        # Assert
        assert all(item["family"] != "parametric" for item in items)

    def test_menu_items_exclude_families_filter_any_brunner_munzel_item(self):
        """Test filtering by exclude_families."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx, exclude_families=["parametric"])
        # Assert
        assert any(item["id"] == "brunner_munzel" for item in items)

    def test_menu_items_exclude_families_filter_any_ttest_ind_item(self):
        """Test filtering by exclude_families."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx, exclude_families=["parametric"])
        # Assert
        assert not any(item["id"] == "ttest_ind" for item in items)

    def test_menu_items_has_brunner_munzel_first(self):
        """Test that Brunner-Munzel appears first for 2-group comparison."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        items = get_menu_items(ctx)
        enabled_items = [i for i in items if i["enabled"]]
        # Assert
        if enabled_items:
            assert enabled_items[0]["id"] == "brunner_munzel"


class TestRecommendTests:
    """Tests for recommend_tests() function."""

    def test_returns_list_of_strings_recommended(self):
        """Test that recommend_tests returns list of test names."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert isinstance(recommended, list)

    def test_returns_list_of_strings_all_name_str_recommended(self):
        """Test that recommend_tests returns list of test names."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert all(isinstance(name, str) for name in recommended)

    def test_brunner_munzel_recommended_first_for_two_groups_case_1(self):
        """Test that Brunner-Munzel is recommended first for 2-group comparison."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert len(recommended) > 0

    def test_brunner_munzel_recommended_first_for_two_groups_case_2(self):
        """Test that Brunner-Munzel is recommended first for 2-group comparison."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert recommended[0] == "brunner_munzel"

    def test_recommend_tests_respects_top_k(self):
        """Test that recommend_tests returns at most top_k tests."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=2)
        # Assert
        assert len(recommended) <= 2

    def test_recommend_tests_sorted_by_priority(self):
        """Test that recommended tests are sorted by priority."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=5)
        priorities = [TEST_RULES[name].priority for name in recommended]
        # Assert
        assert priorities == sorted(priorities, reverse=True)

    def test_recommend_tests_only_applicable_brunner_munzel_recommended(self):
        """Test that only applicable tests are recommended."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        # Act
        recommended = recommend_tests(ctx, top_k=10)
        # Assert
        assert "brunner_munzel" in recommended

    def test_recommend_tests_only_applicable_ttest_ind_recommended(self):
        """Test that only applicable tests are recommended."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        # Act
        recommended = recommend_tests(ctx, top_k=10)
        # Assert
        assert "ttest_ind" not in recommended

    def test_recommend_tests_with_families_filter_all_family_nonparametric_name(self):
        """Test filtering by families parameter."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=5, families=["nonparametric"])
        # Assert
        assert all(TEST_RULES[name].family == "nonparametric" for name in recommended)

    def test_recommend_tests_with_families_filter_brunner_munzel_recommended(self):
        """Test filtering by families parameter."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=5, families=["nonparametric"])
        # Assert
        assert "brunner_munzel" in recommended

    def test_recommend_tests_with_families_filter_ttest_ind_recommended(self):
        """Test filtering by families parameter."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=5, families=["nonparametric"])
        # Assert
        assert "ttest_ind" not in recommended

    def test_recommend_tests_for_paired_design_all_supports_paired_name_recommended(self):
        """Test recommendations for paired design."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert all(TEST_RULES[name].supports_paired for name in recommended)

    def test_recommend_tests_for_paired_design_any_name_recommended_ttest(self):
        """Test recommendations for paired design."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert any(name in ["ttest_rel", "wilcoxon"] for name in recommended)

    def test_recommend_tests_for_three_groups_all_min_groups_name_recommended(self):
        """Test recommendations for three groups."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert all(TEST_RULES[name].min_groups <= 3 for name in recommended)

    def test_recommend_tests_for_three_groups_brunner_munzel_recommended(self):
        """Test recommendations for three groups."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(ctx, top_k=3)
        # Assert
        assert "brunner_munzel" not in recommended

    def test_recommend_tests_empty_when_none_applicable(self):
        """Test that empty list is returned when no tests are applicable."""
        # Arrange
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[10],
            outcome_type="continuous",
            design="between",
        )
        # Act
        recommended = recommend_tests(
            ctx, top_k=3, families=["parametric", "nonparametric"]
        )
        two_group_tests = [
            name for name in recommended if TEST_RULES[name].min_groups > 1
        ]
        # Assert
        assert len(two_group_tests) == 0


class TestRecommendEffectSizes:
    """Tests for recommend_effect_sizes() function."""

    def test_returns_list_of_effect_sizes_effect_sizes(self):
        """Test that recommend_effect_sizes returns effect size names."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        effect_sizes = recommend_effect_sizes(ctx, top_k=3)
        # Assert
        assert isinstance(effect_sizes, list)

    def test_returns_list_of_effect_sizes_all_name_str_effect_sizes(self):
        """Test that recommend_effect_sizes returns effect size names."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        effect_sizes = recommend_effect_sizes(ctx, top_k=3)
        # Assert
        assert all(isinstance(name, str) for name in effect_sizes)

    def test_returns_list_of_effect_sizes_all_family_size_name(self):
        """Test that recommend_effect_sizes returns effect size names."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        effect_sizes = recommend_effect_sizes(ctx, top_k=3)
        # Assert
        assert all(TEST_RULES[name].family == "effect_size" for name in effect_sizes)

    def test_recommend_cohens_d_for_two_groups(self):
        """Test that Cohen's d is recommended for 2-group comparison."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        effect_sizes = recommend_effect_sizes(ctx, top_k=3)
        # Assert
        assert "cohens_d_ind" in effect_sizes

    def test_recommend_cohens_d_paired_for_within_design(self):
        """Test that Cohen's d paired is recommended for within design."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )
        # Act
        effect_sizes = recommend_effect_sizes(ctx, top_k=3)
        # Assert
        assert "cohens_d_paired" in effect_sizes


class TestRecommendPosthoc:
    """Tests for recommend_posthoc() function."""

    def test_returns_list_of_posthoc_tests_case_1(self):
        """Test that recommend_posthoc returns post-hoc test names."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        # Act
        posthoc = recommend_posthoc(ctx, top_k=2)
        # Assert
        assert isinstance(posthoc, list)

    def test_returns_list_of_posthoc_tests_all_name_str(self):
        """Test that recommend_posthoc returns post-hoc test names."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        # Act
        posthoc = recommend_posthoc(ctx, top_k=2)
        # Assert
        assert all(isinstance(name, str) for name in posthoc)

    def test_returns_list_of_posthoc_tests_all_family_name_test_rules(self):
        """Test that recommend_posthoc returns post-hoc test names."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        # Act
        posthoc = recommend_posthoc(ctx, top_k=2)
        # Assert
        assert all(TEST_RULES[name].family == "posthoc" for name in posthoc)

    def test_recommend_tukey_for_three_groups(self):
        """Test that Tukey HSD is recommended for 3+ groups."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        # Act
        posthoc = recommend_posthoc(ctx, top_k=3)
        # Assert
        assert any(name in ["tukey_hsd", "games_howell"] for name in posthoc)

    def test_recommend_games_howell_without_equal_variance_posthoc(self):
        """Test that Games-Howell is recommended without equal variance."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=False,
        )
        # Act
        posthoc = recommend_posthoc(ctx, top_k=3)
        # Assert
        assert "games_howell" in posthoc

    def test_recommend_games_howell_without_equal_variance_tukey_hsd_posthoc(self):
        """Test that Games-Howell is recommended without equal variance."""
        # Arrange
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=False,
        )
        # Act
        posthoc = recommend_posthoc(ctx, top_k=3)
        # Assert
        assert "tukey_hsd" not in posthoc

    def test_no_posthoc_for_two_groups(self):
        """Test that no post-hoc tests are recommended for 2 groups."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        # Act
        posthoc = recommend_posthoc(ctx, top_k=3)
        # Assert
        assert len(posthoc) == 0


class TestTooltipsAndReasons:
    """Tests for tooltip generation and reason messages."""

    def test_tooltip_contains_specific_reason_ok(self):
        """Test that tooltip contains specific reason for rejection."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False
        tooltip = "; ".join(reasons)

    def test_tooltip_contains_specific_reason_normality_lower(self):
        """Test that tooltip contains specific reason for rejection."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]
        ok, reasons = check_applicable(rule, ctx)
        # Act
        tooltip = "; ".join(reasons)
        # Assert
        assert "normality" in tooltip.lower()

    def test_tooltip_multiple_reasons_joined_ok(self):
        """Test that multiple reasons are joined with semicolons."""
        # Arrange
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[5],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert ok is False
        tooltip = "; ".join(reasons)

    def test_tooltip_multiple_reasons_joined_case_2(self):
        """Test that multiple reasons are joined with semicolons."""
        # Arrange
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[5],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        # Act
        ok, reasons = check_applicable(rule, ctx)
        # Assert
        assert len(reasons) >= 2
        tooltip = "; ".join(reasons)

    def test_tooltip_multiple_reasons_joined_case_3(self):
        """Test that multiple reasons are joined with semicolons."""
        # Arrange
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[5],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]
        ok, reasons = check_applicable(rule, ctx)
        # Act
        tooltip = "; ".join(reasons)
        # Assert
        assert ";" in tooltip

    def test_no_tooltip_when_enabled_bm_item(self):
        """Test that tooltip is None when test is enabled."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        items = get_menu_items(ctx)
        # Act
        bm_item = next(i for i in items if i["id"] == "brunner_munzel")
        # Assert
        assert bm_item["enabled"] is True

    def test_no_tooltip_when_enabled_bm_item_2(self):
        """Test that tooltip is None when test is enabled."""
        # Arrange
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        items = get_menu_items(ctx)
        # Act
        bm_item = next(i for i in items if i["id"] == "brunner_munzel")
        # Assert
        assert bm_item["tooltip"] is None


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_selector.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_selector.py
#
# """
# Test Selector - Automatic statistical test selection engine.
#
# This module provides the core logic for determining which statistical tests
# are applicable to a given context, generating UI menu items, and recommending
# tests based on priority.
#
# Key Functions:
# - check_applicable(): Check if a test is applicable to a context
# - get_menu_items(): Generate UI menu items for right-click menus
# - recommend_tests(): Get recommended tests sorted by priority
# - run_all_applicable_tests(): Run all applicable tests in parallel
#
# The Brunner-Munzel test is the recommended default for 2-group comparisons
# due to its robustness (no normality or equal variance assumptions).
# """
#
# from __future__ import annotations
#
# from concurrent.futures import ThreadPoolExecutor
# from typing import Dict, List, Optional, Tuple, Callable, Any
#
# from ._context import StatContext
# from ._rules import TestRule, TEST_RULES, TestFamily
#
#
# # =============================================================================
# # Pretty Labels for UI
# # =============================================================================
#
# _PRETTY_LABELS: Dict[str, str] = {
#     # Parametric
#     "ttest_ind": "t-test (independent)",
#     "ttest_rel": "t-test (paired)",
#     "anova_oneway": "One-way ANOVA",
#     "anova_rm_oneway": "Repeated-measures ANOVA",
#     "anova_twoway": "Two-way ANOVA",
#     "anova_twoway_mixed": "Mixed-design ANOVA",
#     "welch_anova": "Welch's ANOVA",
#
#     # Nonparametric
#     "brunner_munzel": "Brunner-Munzel test (recommended)",
#     "mannwhitneyu": "Mann-Whitney U",
#     "wilcoxon": "Wilcoxon signed-rank",
#     "kruskal": "Kruskal-Wallis",
#     "friedman": "Friedman test",
#
#     # Categorical
#     "chi2_independence": "Chi-square test",
#     "fisher_exact": "Fisher's exact test",
#     "mcnemar": "McNemar's test",
#
#     # Correlation
#     "pearsonr": "Pearson correlation",
#     "spearmanr": "Spearman correlation",
#
#     # Normality/Other
#     "shapiro": "Shapiro-Wilk (normality)",
#     "levene": "Levene's test (variance)",
#
#     # Posthoc
#     "tukey_hsd": "Tukey HSD",
#     "dunnett": "Dunnett's test (vs control)",
#     "games_howell": "Games-Howell",
#
#     # Effect sizes
#     "cohens_d_ind": "Cohen's d (independent)",
#     "cohens_d_paired": "Cohen's d (paired)",
#     "hedges_g": "Hedges' g",
#     "cliffs_delta": "Cliff's delta",
#     "eta_squared": "Eta-squared (eta^2)",
#     "partial_eta_squared": "Partial eta-squared",
#     "effect_size_r": "Effect size r",
#     "odds_ratio": "Odds ratio",
#     "risk_ratio": "Risk ratio",
#     "prob_superiority": "P(X>Y) superiority",
# }
#
#
# def _pretty_label(name: str) -> str:
#     """Get human-readable label for a test name."""
#     return _PRETTY_LABELS.get(name, name)
#
#
# # =============================================================================
# # Core Applicability Check
# # =============================================================================
#
#
# def check_applicable(
#     rule: TestRule,
#     ctx: StatContext,
# ) -> Tuple[bool, List[str]]:
#     """
#     Check whether a given statistical test is applicable to the context.
#
#     This function evaluates all conditions in the TestRule against the
#     StatContext and returns both the result and human-readable reasons
#     for any failures (suitable for tooltips).
#
#     Parameters
#     ----------
#     rule : TestRule
#         The rule definition for a specific test.
#     ctx : StatContext
#         The context inferred from the figure and data.
#
#     Returns
#     -------
#     ok : bool
#         True if applicable, False otherwise.
#     reasons : list of str
#         If not applicable, human-readable reasons for tooltips.
#
#     Examples
#     --------
#     >>> from scitex_stats.auto import StatContext, TEST_RULES, check_applicable
#     >>> ctx = StatContext(
#     ...     n_groups=2,
#     ...     sample_sizes=[30, 32],
#     ...     outcome_type="continuous",
#     ...     design="between",
#     ...     paired=False,
#     ...     has_control_group=False,
#     ...     n_factors=1
#     ... )
#     >>> rule = TEST_RULES["ttest_ind"]
#     >>> ok, reasons = check_applicable(rule, ctx)
#     >>> ok
#     True
#
#     >>> ctx.normality_ok = False
#     >>> ok, reasons = check_applicable(rule, ctx)
#     >>> ok
#     False
#     >>> "normality" in reasons[0].lower()
#     True
#     """
#     reasons: List[str] = []
#
#     # Number of groups
#     if ctx.n_groups < rule.min_groups:
#         reasons.append(
#             f"Requires at least {rule.min_groups} groups "
#             f"(current: {ctx.n_groups})"
#         )
#     if rule.max_groups is not None and ctx.n_groups > rule.max_groups:
#         reasons.append(
#             f"Maximum {rule.max_groups} groups allowed "
#             f"(current: {ctx.n_groups})"
#         )
#
#     # Outcome type
#     if ctx.outcome_type not in rule.outcome_types:
#         allowed = ", ".join(sorted(rule.outcome_types))
#         reasons.append(
#             f"This test is for {allowed} data "
#             f"(current: {ctx.outcome_type})"
#         )
#
#     # Paired / unpaired
#     effective_paired = ctx.effective_paired
#     if effective_paired is True and not rule.supports_paired:
#         reasons.append("This test does not support paired/repeated measures")
#     if effective_paired is False and not rule.supports_unpaired:
#         reasons.append("This test does not support independent groups")
#
#     # Design
#     if ctx.design not in rule.design_allowed:
#         allowed = ", ".join(sorted(rule.design_allowed))
#         reasons.append(
#             f"Design '{ctx.design}' not supported "
#             f"(allowed: {allowed})"
#         )
#
#     # Sample sizes
#     if rule.min_n_total is not None:
#         n_total = ctx.n_total
#         if n_total < rule.min_n_total:
#             reasons.append(
#                 f"Sample size too small (need >= {rule.min_n_total}, "
#                 f"current: {n_total})"
#             )
#
#     if rule.min_n_per_group is not None:
#         min_n = ctx.min_n_per_group
#         if min_n < rule.min_n_per_group:
#             reasons.append(
#                 f"Each group needs n >= {rule.min_n_per_group} "
#                 f"(smallest group: {min_n})"
#             )
#
#     # Normality assumption
#     if rule.needs_normality and ctx.normality_ok is False:
#         reasons.append(
#             "Normality assumption not met (consider nonparametric test)"
#         )
#
#     # Equal variance assumption
#     if rule.needs_equal_variance and ctx.variance_homogeneity_ok is False:
#         reasons.append(
#             "Equal variance assumption not met (consider Welch or nonparametric)"
#         )
#
#     # Control group requirement
#     if rule.requires_control_group and not ctx.has_control_group:
#         reasons.append("This test requires a designated control group")
#
#     # Factor constraints
#     if rule.min_factors is not None and ctx.n_factors < rule.min_factors:
#         reasons.append(
#             f"Requires at least {rule.min_factors} factor(s) "
#             f"(current: {ctx.n_factors})"
#         )
#     if rule.max_factors is not None and ctx.n_factors > rule.max_factors:
#         reasons.append(
#             f"Maximum {rule.max_factors} factor(s) allowed "
#             f"(current: {ctx.n_factors})"
#         )
#
#     ok = len(reasons) == 0
#     return ok, reasons
#
#
# # =============================================================================
# # Menu Generation
# # =============================================================================
#
#
# def get_menu_items(
#     ctx: StatContext,
#     include_families: Optional[List[TestFamily]] = None,
#     exclude_families: Optional[List[TestFamily]] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Build UI menu items for the given statistical context.
#
#     Returns a list of menu item dictionaries suitable for right-click
#     context menus. Enabled items are sorted to the top, then by priority.
#
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     include_families : list of TestFamily or None
#         If provided, only tests whose family is in this list will be considered.
#     exclude_families : list of TestFamily or None
#         If provided, tests whose family is in this list will be skipped.
#
#     Returns
#     -------
#     items : list of dict
#         Each item has:
#         - id (str): internal test name
#         - label (str): human-readable label
#         - family (str): test family
#         - enabled (bool): whether this test is applicable
#         - tooltip (str or None): reason why disabled (if any)
#         - priority (int): for sorting/recommendation
#
#     Examples
#     --------
#     >>> ctx = StatContext(
#     ...     n_groups=2,
#     ...     sample_sizes=[30, 32],
#     ...     outcome_type="continuous",
#     ...     design="between",
#     ...     paired=False,
#     ...     has_control_group=False,
#     ...     n_factors=1
#     ... )
#     >>> items = get_menu_items(ctx)
#     >>> enabled_items = [i for i in items if i["enabled"]]
#     >>> len(enabled_items) > 0
#     True
#     """
#     items: List[Dict[str, Any]] = []
#     include_set = set(include_families or [])
#     exclude_set = set(exclude_families or [])
#
#     for name, rule in TEST_RULES.items():
#         # Family-based filtering
#         if include_set and rule.family not in include_set:
#             continue
#         if rule.family in exclude_set:
#             continue
#
#         ok, reasons = check_applicable(rule, ctx)
#         tooltip = None if ok else "; ".join(reasons)
#
#         items.append({
#             "id": name,
#             "label": _pretty_label(name),
#             "family": rule.family,
#             "enabled": ok,
#             "tooltip": tooltip,
#             "priority": rule.priority,
#         })
#
#     # Sort: enabled first, then by priority (desc), then label
#     items.sort(
#         key=lambda d: (
#             not d["enabled"],      # False (enabled) -> 0 -> top
#             -int(d["priority"]),
#             d["label"],
#         )
#     )
#     return items
#
#
# # =============================================================================
# # Test Recommendation
# # =============================================================================
#
#
# def recommend_tests(
#     ctx: StatContext,
#     top_k: int = 3,
#     families: Optional[List[TestFamily]] = None,
# ) -> List[str]:
#     """
#     Recommend tests for the given context.
#
#     Returns test names sorted by priority. Brunner-Munzel is the
#     recommended default for 2-group comparisons (priority 110).
#
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     top_k : int
#         Number of top tests to return.
#     families : list of TestFamily or None
#         Families to consider. If None, uses standard test families
#         (parametric, nonparametric, categorical, correlation).
#
#     Returns
#     -------
#     test_names : list of str
#         Internal names of recommended tests, sorted by priority.
#
#     Examples
#     --------
#     >>> ctx = StatContext(
#     ...     n_groups=2,
#     ...     sample_sizes=[30, 32],
#     ...     outcome_type="continuous",
#     ...     design="between",
#     ...     paired=False,
#     ...     has_control_group=False,
#     ...     n_factors=1
#     ... )
#     >>> recommended = recommend_tests(ctx, top_k=3)
#     >>> "brunner_munzel" in recommended
#     True
#     """
#     if families is None:
#         families = [
#             "parametric",
#             "nonparametric",
#             "categorical",
#             "correlation",
#         ]
#
#     families_set = set(families)
#     candidates: List[Tuple[int, str]] = []
#
#     for name, rule in TEST_RULES.items():
#         if rule.family not in families_set:
#             continue
#
#         ok, _ = check_applicable(rule, ctx)
#         if not ok:
#             continue
#
#         candidates.append((rule.priority, name))
#
#     # Sort by priority (high -> first)
#     candidates.sort(reverse=True)
#
#     return [name for _, name in candidates[:top_k]]
#
#
# def recommend_effect_sizes(
#     ctx: StatContext,
#     top_k: int = 3,
# ) -> List[str]:
#     """
#     Recommend effect size measures for the given context.
#
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     top_k : int
#         Number of top effect sizes to return.
#
#     Returns
#     -------
#     effect_names : list of str
#         Internal names of recommended effect sizes.
#     """
#     return recommend_tests(ctx, top_k=top_k, families=["effect_size"])
#
#
# def recommend_posthoc(
#     ctx: StatContext,
#     top_k: int = 2,
# ) -> List[str]:
#     """
#     Recommend post-hoc tests for the given context.
#
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     top_k : int
#         Number of top post-hoc tests to return.
#
#     Returns
#     -------
#     posthoc_names : list of str
#         Internal names of recommended post-hoc tests.
#     """
#     return recommend_tests(ctx, top_k=top_k, families=["posthoc"])
#
#
# # =============================================================================
# # Parallel Test Execution
# # =============================================================================
#
#
# def run_all_applicable_tests(
#     ctx: StatContext,
#     data: Any,
#     test_backend: Dict[str, Callable],
#     families: Optional[List[TestFamily]] = None,
#     max_workers: Optional[int] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Run all applicable statistical tests in parallel.
#
#     Executes all tests that pass check_applicable() using a thread pool,
#     and returns results sorted by priority.
#
#     Parameters
#     ----------
#     ctx : StatContext
#         Statistical context.
#     data : Any
#         Data to pass to test functions (typically StatData or similar).
#     test_backend : dict
#         Dictionary mapping test names to callable functions.
#         Each function should accept data and return a result dict.
#     families : list of TestFamily or None
#         Families to include. Defaults to standard test families.
#     max_workers : int or None
#         Maximum number of parallel workers. None uses default.
#
#     Returns
#     -------
#     results : list of dict
#         Test results sorted by priority (highest first).
#         Each result includes at least 'test_name' key.
#
#     Examples
#     --------
#     >>> # Define test backends
#     >>> def ttest_backend(data):
#     ...     from scipy import stats
#     ...     stat, p = stats.ttest_ind(data.group1, data.group2)
#     ...     return {"test_name": "ttest_ind", "stat": stat, "p_raw": p}
#     >>>
#     >>> backends = {"ttest_ind": ttest_backend}
#     >>> # results = run_all_applicable_tests(ctx, data, backends)
#     """
#     if families is None:
#         families = [
#             "parametric",
#             "nonparametric",
#             "categorical",
#             "correlation",
#         ]
#
#     families_set = set(families)
#     tasks: List[Tuple[str, int]] = []
#
#     # Find applicable tests
#     for name, rule in TEST_RULES.items():
#         if rule.family not in families_set:
#             continue
#
#         ok, _ = check_applicable(rule, ctx)
#         if not ok:
#             continue
#
#         if name not in test_backend:
#             continue
#
#         tasks.append((name, rule.priority))
#
#     results: List[Dict[str, Any]] = []
#
#     def run_single(name: str) -> Dict[str, Any]:
#         """Run a single test and handle errors."""
#         try:
#             return test_backend[name](data)
#         except Exception as e:
#             return {
#                 "test_name": name,
#                 "p_raw": None,
#                 "stat": None,
#                 "error": str(e),
#             }
#
#     # Run in parallel
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(run_single, name): (name, priority)
#             for name, priority in tasks
#         }
#         for future in futures:
#             result = future.result()
#             results.append(result)
#
#     # Sort by priority (high -> first)
#     def get_priority(r: Dict) -> int:
#         test_name = r.get("test_name", "")
#         rule = TEST_RULES.get(test_name)
#         return rule.priority if rule else 0
#
#     results.sort(key=get_priority, reverse=True)
#     return results
#
#
# # =============================================================================
# # Public API
# # =============================================================================
#
# __all__ = [
#     "check_applicable",
#     "get_menu_items",
#     "recommend_tests",
#     "recommend_effect_sizes",
#     "recommend_posthoc",
#     "run_all_applicable_tests",
# ]
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_selector.py
# --------------------------------------------------------------------------------


# ============================================================
# run_all_applicable_tests + _pretty_label
# ============================================================


class TestRunAllApplicableTests:
    """Parallel-dispatch entrypoint for the recommender."""

    def test_runs_only_applicable_backends_results_list(self):
        # Arrange
        from scitex_stats.auto import StatContext
        from scitex_stats.auto._selector import run_all_applicable_tests
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="between",
            paired=False,
        )
        backends = {
            "ttest_ind": lambda d: {"test_name": "ttest_ind", "stat": 1.0, "p_raw": 0.04},
            "mannwhitneyu": lambda d: {"test_name": "mannwhitneyu", "stat": 50, "p_raw": 0.05},
        }
        # Act
        results = run_all_applicable_tests(ctx, data=None, test_backend=backends)
        # Assert
        assert isinstance(results, list) and results
        names = {r["test_name"] for r in results}

    def test_runs_only_applicable_backends_names_set_keys(self):
        # Arrange
        from scitex_stats.auto import StatContext
        from scitex_stats.auto._selector import run_all_applicable_tests
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="between",
            paired=False,
        )
        backends = {
            "ttest_ind": lambda d: {"test_name": "ttest_ind", "stat": 1.0, "p_raw": 0.04},
            "mannwhitneyu": lambda d: {"test_name": "mannwhitneyu", "stat": 50, "p_raw": 0.05},
        }
        # Act
        results = run_all_applicable_tests(ctx, data=None, test_backend=backends)
        names = {r["test_name"] for r in results}
        # Assert
        assert names & set(backends.keys())

    def test_errors_in_backend_become_result_entries_results(self):
        # Arrange
        from scitex_stats.auto import StatContext
        from scitex_stats.auto._selector import run_all_applicable_tests
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="between",
            paired=False,
        )
        def _boom(_data):
            raise RuntimeError("boom")
        backends = {"ttest_ind": _boom}
        # Act
        results = run_all_applicable_tests(ctx, data=None, test_backend=backends)
        # Assert
        assert results
        errored = next(r for r in results if r["test_name"] == "ttest_ind")

    def test_errors_in_backend_become_result_entries_boom_errored_error(self):
        # Arrange
        from scitex_stats.auto import StatContext
        from scitex_stats.auto._selector import run_all_applicable_tests
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="between",
            paired=False,
        )
        def _boom(_data):
            raise RuntimeError("boom")
        backends = {"ttest_ind": _boom}
        results = run_all_applicable_tests(ctx, data=None, test_backend=backends)
        # Act
        errored = next(r for r in results if r["test_name"] == "ttest_ind")
        # Assert
        assert errored["error"] == "boom"

    def test_results_sorted_by_priority_descending(self):
        # Arrange
        from scitex_stats.auto import StatContext
        from scitex_stats.auto._selector import run_all_applicable_tests
        from scitex_stats.auto._rules import TEST_RULES
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="between",
            paired=False,
        )
        backends = {
            name: (lambda d, n=name: {"test_name": n, "stat": 0.0, "p_raw": 0.5})
            for name in ["ttest_ind", "mannwhitneyu", "brunner_munzel"]
        }
        # Act
        results = run_all_applicable_tests(ctx, data=None, test_backend=backends)
        # Assert
        if len(results) >= 2:
            priorities = [
                (TEST_RULES.get(r["test_name"]).priority if TEST_RULES.get(r["test_name"]) else 0)
                for r in results
            ]
            assert priorities == sorted(priorities, reverse=True)


def test_pretty_label_known_name_returns_canonical_form():
    # Arrange
    from scitex_stats.auto._selector import _pretty_label, _PRETTY_LABELS
    # Act
    # Assert
    if _PRETTY_LABELS:
        any_known = next(iter(_PRETTY_LABELS))
        assert _pretty_label(any_known) == _PRETTY_LABELS[any_known]


def test_pretty_label_unknown_name_returns_raw_input():
    # Arrange
    from scitex_stats.auto._selector import _pretty_label
    # Act
    # Assert
    assert _pretty_label("definitely_unknown_test_name") == "definitely_unknown_test_name"
