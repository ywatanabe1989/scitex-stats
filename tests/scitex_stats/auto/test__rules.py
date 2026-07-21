#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for TestRule dataclass and TEST_RULES registry.

Tests cover:
- TestRule dataclass creation and field validation
- TEST_RULES registry completeness and consistency
- get_test_rule() function
- list_tests_by_family() function
- Priority ordering (brunner_munzel should be highest at 110)
- Family categorization and test coverage
- Test rule attributes for specific tests
"""

import pytest

from scitex_stats.auto._rules import (
    TEST_RULES,
    TestRule,
    get_test_rule,
    list_tests_by_family,
)


class TestTestRuleDataclass:
    """Tests for TestRule dataclass."""

    def test_testrule_creation_name_test_example_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.name == "test_example"

    def test_testrule_creation_family_parametric_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.family == "parametric"

    def test_testrule_creation_min_groups_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.min_groups == 2

    def test_testrule_creation_max_groups_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.max_groups == 2

    def test_testrule_creation_outcome_types_rule_continuous(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.outcome_types == {"continuous"}

    def test_testrule_creation_supports_paired_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.supports_paired is False

    def test_testrule_creation_supports_unpaired_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.supports_unpaired is True

    def test_testrule_creation_design_allowed_rule_between(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.design_allowed == {"between"}

    def test_testrule_creation_requires_control_group_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.requires_control_group is False

    def test_testrule_creation_min_n_total_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.min_n_total == 4

    def test_testrule_creation_min_n_per_group_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.min_n_per_group == 2

    def test_testrule_creation_needs_normality_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.needs_normality is True

    def test_testrule_creation_needs_equal_variance_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.needs_equal_variance is False

    def test_testrule_creation_min_factors_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.min_factors == 1

    def test_testrule_creation_max_factors_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.max_factors == 1

    def test_testrule_creation_priority_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.priority == 90

    def test_testrule_creation_description_example_test_rule(self):
        """Test basic TestRule instantiation."""
        # Arrange
        # Act
        rule = TestRule(
            name="test_example",
            family="parametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=True,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )
        # Assert
        assert rule.description == "Example test rule"

    def test_testrule_defaults_priority_rule(self):
        """Test TestRule with default values."""
        # Arrange
        # Act
        rule = TestRule(
            name="minimal",
            family="parametric",
            min_groups=2,
            max_groups=None,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=None,
            min_n_per_group=None,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )
        # Assert
        assert rule.priority == 0

    def test_testrule_defaults_description_rule(self):
        """Test TestRule with default values."""
        # Arrange
        # Act
        rule = TestRule(
            name="minimal",
            family="parametric",
            min_groups=2,
            max_groups=None,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=None,
            min_n_per_group=None,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )
        # Assert
        assert rule.description == ""

    def test_testrule_with_multiple_outcome_types_continuous_outcome_types_rule(self):
        """Test TestRule accepting multiple outcome types."""
        # Arrange
        # Act
        rule = TestRule(
            name="multi_outcome",
            family="nonparametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous", "ordinal"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
        )
        # Assert
        assert "continuous" in rule.outcome_types

    def test_testrule_with_multiple_outcome_types_ordinal_outcome_types_rule(self):
        """Test TestRule accepting multiple outcome types."""
        # Arrange
        # Act
        rule = TestRule(
            name="multi_outcome",
            family="nonparametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous", "ordinal"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
        )
        # Assert
        assert "ordinal" in rule.outcome_types

    def test_testrule_with_multiple_outcome_types_outcome_types_rule(self):
        """Test TestRule accepting multiple outcome types."""
        # Arrange
        # Act
        rule = TestRule(
            name="multi_outcome",
            family="nonparametric",
            min_groups=2,
            max_groups=2,
            outcome_types={"continuous", "ordinal"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=4,
            min_n_per_group=2,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
        )
        # Assert
        assert len(rule.outcome_types) == 2

    def test_testrule_with_multiple_designs_design_allowed_rule(self):
        """Test TestRule with multiple allowed designs."""
        # Arrange
        # Act
        rule = TestRule(
            name="multi_design",
            family="correlation",
            min_groups=1,
            max_groups=1,
            outcome_types={"continuous"},
            supports_paired=True,
            supports_unpaired=True,
            design_allowed={"between", "within", "mixed"},
            requires_control_group=False,
            min_n_total=3,
            min_n_per_group=None,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )
        # Assert
        assert len(rule.design_allowed) == 3

    def test_testrule_with_multiple_designs_between_design_allowed_rule(self):
        """Test TestRule with multiple allowed designs."""
        # Arrange
        # Act
        rule = TestRule(
            name="multi_design",
            family="correlation",
            min_groups=1,
            max_groups=1,
            outcome_types={"continuous"},
            supports_paired=True,
            supports_unpaired=True,
            design_allowed={"between", "within", "mixed"},
            requires_control_group=False,
            min_n_total=3,
            min_n_per_group=None,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )
        # Assert
        assert "between" in rule.design_allowed

    def test_testrule_with_multiple_designs_within_design_allowed_rule(self):
        """Test TestRule with multiple allowed designs."""
        # Arrange
        # Act
        rule = TestRule(
            name="multi_design",
            family="correlation",
            min_groups=1,
            max_groups=1,
            outcome_types={"continuous"},
            supports_paired=True,
            supports_unpaired=True,
            design_allowed={"between", "within", "mixed"},
            requires_control_group=False,
            min_n_total=3,
            min_n_per_group=None,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )
        # Assert
        assert "within" in rule.design_allowed

    def test_testrule_with_multiple_designs_mixed_design_allowed_rule(self):
        """Test TestRule with multiple allowed designs."""
        # Arrange
        # Act
        rule = TestRule(
            name="multi_design",
            family="correlation",
            min_groups=1,
            max_groups=1,
            outcome_types={"continuous"},
            supports_paired=True,
            supports_unpaired=True,
            design_allowed={"between", "within", "mixed"},
            requires_control_group=False,
            min_n_total=3,
            min_n_per_group=None,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )
        # Assert
        assert "mixed" in rule.design_allowed


class TestTESTRULESRegistry:
    """Tests for the TEST_RULES registry."""

    def test_registry_is_dict(self):
        """Test that TEST_RULES is a dictionary."""
        # Arrange
        # Act
        # Assert
        assert isinstance(TEST_RULES, dict)

    def test_registry_not_empty(self):
        """Test that TEST_RULES contains tests."""
        # Arrange
        # Act
        # Assert
        assert len(TEST_RULES) > 0

    def test_registry_has_expected_tests(self):
        """Test that registry contains expected test names."""
        # Arrange
        expected_tests = [
            "ttest_ind",
            "ttest_rel",
            "brunner_munzel",
            "mannwhitneyu",
            "wilcoxon",
            "anova_oneway",
            "kruskal",
            "friedman",
            "chi2_independence",
            "fisher_exact",
            "pearsonr",
            "spearmanr",
        ]
        # Act
        # Assert
        for test_name in expected_tests:
            assert test_name in TEST_RULES, f"{test_name} should be in TEST_RULES"

    def test_all_rules_are_testrule_instances(self):
        """Test that all registry values are TestRule instances."""
        # Arrange
        # Act
        # Assert
        for name, rule in TEST_RULES.items():
            assert isinstance(rule, TestRule), f"{name} should be a TestRule instance"

    def test_all_rules_have_matching_names(self):
        """Test that rule names match their registry keys."""
        # Arrange
        # Act
        # Assert
        for key, rule in TEST_RULES.items():
            assert rule.name == key, f"Rule name '{rule.name}' should match key '{key}'"

    def test_all_rules_have_valid_families(self):
        """Test that all rules have valid family types."""
        # Arrange
        valid_families = {
            "parametric",
            "nonparametric",
            "categorical",
            "correlation",
            "normality",
            "effect_size",
            "posthoc",
            "other",
        }
        # Act
        # Assert
        for name, rule in TEST_RULES.items():
            assert rule.family in valid_families, (
                f"{name} has invalid family: {rule.family}"
            )

    def test_all_rules_have_nonempty_descriptions(self):
        """Test that all rules have non-empty descriptions."""
        # Arrange
        # Act
        # Assert
        for name, rule in TEST_RULES.items():
            assert rule.description, f"{name} should have a description"

    def test_all_rule_descriptions_are_strings(self):
        """Test that all rule descriptions are str instances."""
        # Arrange
        # Act
        # Assert
        for name, rule in TEST_RULES.items():
            assert isinstance(rule.description, str)

    def test_all_rules_have_priorities(self):
        """Test that all rules have priority values."""
        # Arrange
        # Act
        # Assert
        for name, rule in TEST_RULES.items():
            assert isinstance(rule.priority, int), f"{name} priority should be int"

    def test_parametric_tests_present(self):
        """Test that parametric tests are in the registry."""
        # Arrange
        parametric_tests = ["ttest_ind", "ttest_rel", "anova_oneway", "welch_anova"]
        # Act
        # Assert
        for test in parametric_tests:
            assert test in TEST_RULES and TEST_RULES[test].family == "parametric"

    def test_nonparametric_tests_present(self):
        """Test that nonparametric tests are in the registry."""
        # Arrange
        nonparametric_tests = [
            "brunner_munzel",
            "mannwhitneyu",
            "wilcoxon",
            "kruskal",
            "friedman",
        ]
        # Act
        # Assert
        for test in nonparametric_tests:
            assert test in TEST_RULES and TEST_RULES[test].family == "nonparametric"

    def test_categorical_tests_present(self):
        """Test that categorical tests are in the registry."""
        # Arrange
        categorical_tests = ["chi2_independence", "fisher_exact", "mcnemar"]
        # Act
        # Assert
        for test in categorical_tests:
            assert test in TEST_RULES and TEST_RULES[test].family == "categorical"

    def test_correlation_tests_present(self):
        """Test that correlation tests are in the registry."""
        # Arrange
        correlation_tests = ["pearsonr", "spearmanr"]
        # Act
        # Assert
        for test in correlation_tests:
            assert test in TEST_RULES and TEST_RULES[test].family == "correlation"

    def test_posthoc_tests_present(self):
        """Test that post-hoc tests are in the registry."""
        # Arrange
        posthoc_tests = ["tukey_hsd", "dunnett", "games_howell"]
        # Act
        # Assert
        for test in posthoc_tests:
            assert test in TEST_RULES and TEST_RULES[test].family == "posthoc"

    def test_effect_size_tests_present(self):
        """Test that effect size measures are in the registry."""
        # Arrange
        effect_size_tests = [
            "cohens_d_ind",
            "cohens_d_paired",
            "hedges_g",
            "cliffs_delta",
            "eta_squared",
        ]
        # Act
        # Assert
        for test in effect_size_tests:
            assert test in TEST_RULES and TEST_RULES[test].family == "effect_size"


class TestBrunnerMunzelPriority:
    """Tests for Brunner-Munzel test as the recommended default."""

    def test_brunner_munzel_exists(self):
        """Test that brunner_munzel is in the registry."""
        # Arrange
        # Act
        # Assert
        assert "brunner_munzel" in TEST_RULES

    def test_brunner_munzel_priority_is_110(self):
        """Test that Brunner-Munzel has priority 110."""
        # Arrange
        # Act
        # Assert
        assert TEST_RULES["brunner_munzel"].priority == 110

    def test_brunner_munzel_has_highest_priority_max_priority(self):
        """Test that Brunner-Munzel has the highest priority overall."""
        # Arrange
        # Act
        max_priority = max(rule.priority for rule in TEST_RULES.values())
        # Assert
        assert max_priority == 110

    def test_brunner_munzel_has_highest_priority_max_priority_test_rules(self):
        """Test that Brunner-Munzel has the highest priority overall."""
        # Arrange
        # Act
        max_priority = max(rule.priority for rule in TEST_RULES.values())
        # Assert
        assert TEST_RULES["brunner_munzel"].priority == max_priority

    def test_brunner_munzel_is_nonparametric(self):
        """Test that Brunner-Munzel is nonparametric."""
        # Arrange
        # Act
        # Assert
        assert TEST_RULES["brunner_munzel"].family == "nonparametric"

    def test_brunner_munzel_is_for_two_groups_min_groups_rule(self):
        """Test that Brunner-Munzel is for 2-group comparisons."""
        # Arrange
        rule = TEST_RULES["brunner_munzel"]
        # Act
        # Assert
        assert rule.min_groups == 2

    def test_brunner_munzel_is_for_two_groups_max_groups_rule(self):
        """Test that Brunner-Munzel is for 2-group comparisons."""
        # Arrange
        rule = TEST_RULES["brunner_munzel"]
        # Act
        # Assert
        assert rule.max_groups == 2

    def test_brunner_munzel_no_assumptions_needs_normality_rule(self):
        """Test that Brunner-Munzel requires no assumptions."""
        # Arrange
        rule = TEST_RULES["brunner_munzel"]
        # Act
        # Assert
        assert rule.needs_normality is False

    def test_brunner_munzel_no_assumptions_needs_equal_variance_rule(self):
        """Test that Brunner-Munzel requires no assumptions."""
        # Arrange
        rule = TEST_RULES["brunner_munzel"]
        # Act
        # Assert
        assert rule.needs_equal_variance is False

    def test_brunner_munzel_between_design_design_allowed_rule(self):
        """Test that Brunner-Munzel is for between-subjects design."""
        # Arrange
        rule = TEST_RULES["brunner_munzel"]
        # Act
        # Assert
        assert "between" in rule.design_allowed

    def test_brunner_munzel_between_design_supports_unpaired_rule(self):
        """Test that Brunner-Munzel is for between-subjects design."""
        # Arrange
        rule = TEST_RULES["brunner_munzel"]
        # Act
        # Assert
        assert rule.supports_unpaired is True

    def test_brunner_munzel_between_design_supports_paired_rule(self):
        """Test that Brunner-Munzel is for between-subjects design."""
        # Arrange
        rule = TEST_RULES["brunner_munzel"]
        # Act
        # Assert
        assert rule.supports_paired is False


class TestGetTestRule:
    """Tests for get_test_rule() function."""

    def test_get_existing_test_rule(self):
        """Test retrieving an existing test rule."""
        # Arrange
        # Act
        rule = get_test_rule("ttest_ind")
        # Assert
        assert rule is not None

    def test_get_existing_test_name_ttest_ind_rule(self):
        """Test retrieving an existing test rule."""
        # Arrange
        # Act
        rule = get_test_rule("ttest_ind")
        # Assert
        assert rule.name == "ttest_ind"

    def test_get_nonexistent_test(self):
        """Test retrieving a nonexistent test returns None."""
        # Arrange
        # Act
        rule = get_test_rule("nonexistent_test")
        # Assert
        assert rule is None

    def test_get_all_expected_tests(self):
        """Test retrieving all expected tests."""
        # Arrange
        test_names = [
            "ttest_ind",
            "brunner_munzel",
            "anova_oneway",
            "kruskal",
            "chi2_independence",
            "pearsonr",
        ]
        # Act
        # Assert
        for name in test_names:
            rule = get_test_rule(name)
            assert rule is not None and rule.name == name

    def test_get_rule_returns_correct_type(self):
        """Test that get_test_rule returns TestRule instance."""
        # Arrange
        # Act
        rule = get_test_rule("ttest_ind")
        # Assert
        assert isinstance(rule, TestRule)


class TestListTestsByFamily:
    """Tests for list_tests_by_family() function."""

    def test_list_parametric_tests_dict(self):
        """Test listing parametric tests."""
        # Arrange
        # Act
        parametric = list_tests_by_family("parametric")
        # Assert
        assert isinstance(parametric, dict)

    def test_list_parametric_tests_case_2(self):
        """Test listing parametric tests."""
        # Arrange
        # Act
        parametric = list_tests_by_family("parametric")
        # Assert
        assert len(parametric) > 0

    def test_list_parametric_tests_ttest_ind(self):
        """Test listing parametric tests."""
        # Arrange
        # Act
        parametric = list_tests_by_family("parametric")
        # Assert
        assert "ttest_ind" in parametric

    def test_list_parametric_tests_anova_oneway(self):
        """Test listing parametric tests."""
        # Arrange
        # Act
        parametric = list_tests_by_family("parametric")
        # Assert
        assert "anova_oneway" in parametric

    def test_list_parametric_tests_all_family_rule_values(self):
        """Test listing parametric tests."""
        # Arrange
        # Act
        parametric = list_tests_by_family("parametric")
        # Assert
        assert all(rule.family == "parametric" for rule in parametric.values())

    def test_list_nonparametric_tests_dict(self):
        """Test listing nonparametric tests."""
        # Arrange
        # Act
        nonparametric = list_tests_by_family("nonparametric")
        # Assert
        assert isinstance(nonparametric, dict)

    def test_list_nonparametric_tests_case_2(self):
        """Test listing nonparametric tests."""
        # Arrange
        # Act
        nonparametric = list_tests_by_family("nonparametric")
        # Assert
        assert len(nonparametric) > 0

    def test_list_nonparametric_tests_brunner_munzel(self):
        """Test listing nonparametric tests."""
        # Arrange
        # Act
        nonparametric = list_tests_by_family("nonparametric")
        # Assert
        assert "brunner_munzel" in nonparametric

    def test_list_nonparametric_tests_mannwhitneyu(self):
        """Test listing nonparametric tests."""
        # Arrange
        # Act
        nonparametric = list_tests_by_family("nonparametric")
        # Assert
        assert "mannwhitneyu" in nonparametric

    def test_list_nonparametric_tests_kruskal(self):
        """Test listing nonparametric tests."""
        # Arrange
        # Act
        nonparametric = list_tests_by_family("nonparametric")
        # Assert
        assert "kruskal" in nonparametric

    def test_list_nonparametric_tests_all_family_rule_values(self):
        """Test listing nonparametric tests."""
        # Arrange
        # Act
        nonparametric = list_tests_by_family("nonparametric")
        # Assert
        assert all(rule.family == "nonparametric" for rule in nonparametric.values())

    def test_list_categorical_tests_dict(self):
        """Test listing categorical tests."""
        # Arrange
        # Act
        categorical = list_tests_by_family("categorical")
        # Assert
        assert isinstance(categorical, dict)

    def test_list_categorical_tests_case_2(self):
        """Test listing categorical tests."""
        # Arrange
        # Act
        categorical = list_tests_by_family("categorical")
        # Assert
        assert len(categorical) > 0

    def test_list_categorical_tests_chi2_independence(self):
        """Test listing categorical tests."""
        # Arrange
        # Act
        categorical = list_tests_by_family("categorical")
        # Assert
        assert "chi2_independence" in categorical

    def test_list_categorical_tests_fisher_exact(self):
        """Test listing categorical tests."""
        # Arrange
        # Act
        categorical = list_tests_by_family("categorical")
        # Assert
        assert "fisher_exact" in categorical

    def test_list_categorical_tests_all_family_rule_values(self):
        """Test listing categorical tests."""
        # Arrange
        # Act
        categorical = list_tests_by_family("categorical")
        # Assert
        assert all(rule.family == "categorical" for rule in categorical.values())

    def test_list_correlation_tests_dict(self):
        """Test listing correlation tests."""
        # Arrange
        # Act
        correlation = list_tests_by_family("correlation")
        # Assert
        assert isinstance(correlation, dict)

    def test_list_correlation_tests_case_2(self):
        """Test listing correlation tests."""
        # Arrange
        # Act
        correlation = list_tests_by_family("correlation")
        # Assert
        assert len(correlation) > 0

    def test_list_correlation_tests_pearsonr(self):
        """Test listing correlation tests."""
        # Arrange
        # Act
        correlation = list_tests_by_family("correlation")
        # Assert
        assert "pearsonr" in correlation

    def test_list_correlation_tests_spearmanr(self):
        """Test listing correlation tests."""
        # Arrange
        # Act
        correlation = list_tests_by_family("correlation")
        # Assert
        assert "spearmanr" in correlation

    def test_list_correlation_tests_all_family_rule_values(self):
        """Test listing correlation tests."""
        # Arrange
        # Act
        correlation = list_tests_by_family("correlation")
        # Assert
        assert all(rule.family == "correlation" for rule in correlation.values())

    def test_list_posthoc_tests_dict(self):
        """Test listing post-hoc tests."""
        # Arrange
        # Act
        posthoc = list_tests_by_family("posthoc")
        # Assert
        assert isinstance(posthoc, dict)

    def test_list_posthoc_tests_case_2(self):
        """Test listing post-hoc tests."""
        # Arrange
        # Act
        posthoc = list_tests_by_family("posthoc")
        # Assert
        assert len(posthoc) > 0

    def test_list_posthoc_tests_tukey_hsd(self):
        """Test listing post-hoc tests."""
        # Arrange
        # Act
        posthoc = list_tests_by_family("posthoc")
        # Assert
        assert "tukey_hsd" in posthoc

    def test_list_posthoc_tests_dunnett(self):
        """Test listing post-hoc tests."""
        # Arrange
        # Act
        posthoc = list_tests_by_family("posthoc")
        # Assert
        assert "dunnett" in posthoc

    def test_list_posthoc_tests_all_family_rule_values(self):
        """Test listing post-hoc tests."""
        # Arrange
        # Act
        posthoc = list_tests_by_family("posthoc")
        # Assert
        assert all(rule.family == "posthoc" for rule in posthoc.values())

    def test_list_effect_size_tests_effect_size_dict(self):
        """Test listing effect size measures."""
        # Arrange
        # Act
        effect_size = list_tests_by_family("effect_size")
        # Assert
        assert isinstance(effect_size, dict)

    def test_list_effect_size_tests_effect_size(self):
        """Test listing effect size measures."""
        # Arrange
        # Act
        effect_size = list_tests_by_family("effect_size")
        # Assert
        assert len(effect_size) > 0

    def test_list_effect_size_tests_cohens_ind_effect_size(self):
        """Test listing effect size measures."""
        # Arrange
        # Act
        effect_size = list_tests_by_family("effect_size")
        # Assert
        assert "cohens_d_ind" in effect_size

    def test_list_effect_size_tests_eta_squared_effect_size(self):
        """Test listing effect size measures."""
        # Arrange
        # Act
        effect_size = list_tests_by_family("effect_size")
        # Assert
        assert "eta_squared" in effect_size

    def test_list_effect_size_tests_all_family_rule_values(self):
        """Test listing effect size measures."""
        # Arrange
        # Act
        effect_size = list_tests_by_family("effect_size")
        # Assert
        assert all(rule.family == "effect_size" for rule in effect_size.values())

    def test_list_normality_tests_dict(self):
        """Test listing normality tests."""
        # Arrange
        # Act
        normality = list_tests_by_family("normality")
        # Assert
        assert isinstance(normality, dict)

    def test_list_normality_tests_shapiro(self):
        """Test listing normality tests."""
        # Arrange
        # Act
        normality = list_tests_by_family("normality")
        # Assert
        assert "shapiro" in normality

    def test_list_normality_tests_all_family_rule_values(self):
        """Test listing normality tests."""
        # Arrange
        # Act
        normality = list_tests_by_family("normality")
        # Assert
        assert all(rule.family == "normality" for rule in normality.values())

    def test_list_other_tests_dict(self):
        """Test listing other tests."""
        # Arrange
        # Act
        other = list_tests_by_family("other")
        # Assert
        assert isinstance(other, dict)

    def test_list_other_tests_levene(self):
        """Test listing other tests."""
        # Arrange
        # Act
        other = list_tests_by_family("other")
        # Assert
        assert "levene" in other

    def test_list_other_tests_all_family_rule_values(self):
        """Test listing other tests."""
        # Arrange
        # Act
        other = list_tests_by_family("other")
        # Assert
        assert all(rule.family == "other" for rule in other.values())

    def test_list_tests_empty_family(self):
        """Test listing tests from nonexistent family returns empty dict."""
        # Arrange
        # Act
        result = list_tests_by_family("nonexistent_family")
        # Assert
        assert result == {}


class TestSpecificTestRules:
    """Tests for specific test rule attributes."""

    def test_ttest_ind_rule_family_parametric(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert rule.family == "parametric"

    def test_ttest_ind_rule_min_groups(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert rule.min_groups == 2

    def test_ttest_ind_rule_max_groups(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert rule.max_groups == 2

    def test_ttest_ind_rule_continuous_outcome_types(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert "continuous" in rule.outcome_types

    def test_ttest_ind_rule_supports_unpaired(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert rule.supports_unpaired is True

    def test_ttest_ind_rule_supports_paired(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert rule.supports_paired is False

    def test_ttest_ind_rule_between_design_allowed(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert "between" in rule.design_allowed

    def test_ttest_ind_rule_needs_normality(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert rule.needs_normality is True

    def test_ttest_ind_rule_needs_equal_variance(self):
        """Test t-test (independent) rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_ind"]
        # Act
        # Assert
        assert rule.needs_equal_variance is False

    def test_ttest_rel_rule_family_parametric(self):
        """Test paired t-test rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_rel"]
        # Act
        # Assert
        assert rule.family == "parametric"

    def test_ttest_rel_rule_min_groups(self):
        """Test paired t-test rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_rel"]
        # Act
        # Assert
        assert rule.min_groups == 2

    def test_ttest_rel_rule_max_groups(self):
        """Test paired t-test rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_rel"]
        # Act
        # Assert
        assert rule.max_groups == 2

    def test_ttest_rel_rule_supports_paired(self):
        """Test paired t-test rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_rel"]
        # Act
        # Assert
        assert rule.supports_paired is True

    def test_ttest_rel_rule_supports_unpaired(self):
        """Test paired t-test rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_rel"]
        # Act
        # Assert
        assert rule.supports_unpaired is False

    def test_ttest_rel_rule_within_design_allowed(self):
        """Test paired t-test rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_rel"]
        # Act
        # Assert
        assert "within" in rule.design_allowed

    def test_ttest_rel_rule_needs_normality(self):
        """Test paired t-test rule attributes."""
        # Arrange
        rule = TEST_RULES["ttest_rel"]
        # Act
        # Assert
        assert rule.needs_normality is True

    def test_anova_oneway_rule_family_parametric(self):
        """Test one-way ANOVA rule attributes."""
        # Arrange
        rule = TEST_RULES["anova_oneway"]
        # Act
        # Assert
        assert rule.family == "parametric"

    def test_anova_oneway_rule_min_groups(self):
        """Test one-way ANOVA rule attributes."""
        # Arrange
        rule = TEST_RULES["anova_oneway"]
        # Act
        # Assert
        assert rule.min_groups == 3

    def test_anova_oneway_rule_max_groups(self):
        """Test one-way ANOVA rule attributes."""
        # Arrange
        rule = TEST_RULES["anova_oneway"]
        # Act
        # Assert
        assert rule.max_groups is None

    def test_anova_oneway_rule_supports_unpaired(self):
        """Test one-way ANOVA rule attributes."""
        # Arrange
        rule = TEST_RULES["anova_oneway"]
        # Act
        # Assert
        assert rule.supports_unpaired is True

    def test_anova_oneway_rule_between_design_allowed(self):
        """Test one-way ANOVA rule attributes."""
        # Arrange
        rule = TEST_RULES["anova_oneway"]
        # Act
        # Assert
        assert "between" in rule.design_allowed

    def test_anova_oneway_rule_needs_normality(self):
        """Test one-way ANOVA rule attributes."""
        # Arrange
        rule = TEST_RULES["anova_oneway"]
        # Act
        # Assert
        assert rule.needs_normality is True

    def test_anova_oneway_rule_needs_equal_variance(self):
        """Test one-way ANOVA rule attributes."""
        # Arrange
        rule = TEST_RULES["anova_oneway"]
        # Act
        # Assert
        assert rule.needs_equal_variance is True

    def test_mannwhitneyu_rule_family_nonparametric(self):
        """Test Mann-Whitney U rule attributes."""
        # Arrange
        rule = TEST_RULES["mannwhitneyu"]
        # Act
        # Assert
        assert rule.family == "nonparametric"

    def test_mannwhitneyu_rule_min_groups(self):
        """Test Mann-Whitney U rule attributes."""
        # Arrange
        rule = TEST_RULES["mannwhitneyu"]
        # Act
        # Assert
        assert rule.min_groups == 2

    def test_mannwhitneyu_rule_max_groups(self):
        """Test Mann-Whitney U rule attributes."""
        # Arrange
        rule = TEST_RULES["mannwhitneyu"]
        # Act
        # Assert
        assert rule.max_groups == 2

    def test_mannwhitneyu_rule_continuous_outcome_types(self):
        """Test Mann-Whitney U rule attributes."""
        # Arrange
        rule = TEST_RULES["mannwhitneyu"]
        # Act
        # Assert
        assert "continuous" in rule.outcome_types

    def test_mannwhitneyu_rule_ordinal_outcome_types(self):
        """Test Mann-Whitney U rule attributes."""
        # Arrange
        rule = TEST_RULES["mannwhitneyu"]
        # Act
        # Assert
        assert "ordinal" in rule.outcome_types

    def test_mannwhitneyu_rule_needs_normality(self):
        """Test Mann-Whitney U rule attributes."""
        # Arrange
        rule = TEST_RULES["mannwhitneyu"]
        # Act
        # Assert
        assert rule.needs_normality is False

    def test_mannwhitneyu_rule_needs_equal_variance(self):
        """Test Mann-Whitney U rule attributes."""
        # Arrange
        rule = TEST_RULES["mannwhitneyu"]
        # Act
        # Assert
        assert rule.needs_equal_variance is False

    def test_wilcoxon_rule_family_nonparametric(self):
        """Test Wilcoxon signed-rank rule attributes."""
        # Arrange
        rule = TEST_RULES["wilcoxon"]
        # Act
        # Assert
        assert rule.family == "nonparametric"

    def test_wilcoxon_rule_min_groups(self):
        """Test Wilcoxon signed-rank rule attributes."""
        # Arrange
        rule = TEST_RULES["wilcoxon"]
        # Act
        # Assert
        assert rule.min_groups == 2

    def test_wilcoxon_rule_max_groups(self):
        """Test Wilcoxon signed-rank rule attributes."""
        # Arrange
        rule = TEST_RULES["wilcoxon"]
        # Act
        # Assert
        assert rule.max_groups == 2

    def test_wilcoxon_rule_supports_paired(self):
        """Test Wilcoxon signed-rank rule attributes."""
        # Arrange
        rule = TEST_RULES["wilcoxon"]
        # Act
        # Assert
        assert rule.supports_paired is True

    def test_wilcoxon_rule_supports_unpaired(self):
        """Test Wilcoxon signed-rank rule attributes."""
        # Arrange
        rule = TEST_RULES["wilcoxon"]
        # Act
        # Assert
        assert rule.supports_unpaired is False

    def test_wilcoxon_rule_within_design_allowed(self):
        """Test Wilcoxon signed-rank rule attributes."""
        # Arrange
        rule = TEST_RULES["wilcoxon"]
        # Act
        # Assert
        assert "within" in rule.design_allowed

    def test_kruskal_rule_family_nonparametric(self):
        """Test Kruskal-Wallis rule attributes."""
        # Arrange
        rule = TEST_RULES["kruskal"]
        # Act
        # Assert
        assert rule.family == "nonparametric"

    def test_kruskal_rule_min_groups(self):
        """Test Kruskal-Wallis rule attributes."""
        # Arrange
        rule = TEST_RULES["kruskal"]
        # Act
        # Assert
        assert rule.min_groups == 3

    def test_kruskal_rule_max_groups(self):
        """Test Kruskal-Wallis rule attributes."""
        # Arrange
        rule = TEST_RULES["kruskal"]
        # Act
        # Assert
        assert rule.max_groups is None

    def test_kruskal_rule_supports_unpaired(self):
        """Test Kruskal-Wallis rule attributes."""
        # Arrange
        rule = TEST_RULES["kruskal"]
        # Act
        # Assert
        assert rule.supports_unpaired is True

    def test_kruskal_rule_between_design_allowed(self):
        """Test Kruskal-Wallis rule attributes."""
        # Arrange
        rule = TEST_RULES["kruskal"]
        # Act
        # Assert
        assert "between" in rule.design_allowed

    def test_friedman_rule_family_nonparametric(self):
        """Test Friedman rule attributes."""
        # Arrange
        rule = TEST_RULES["friedman"]
        # Act
        # Assert
        assert rule.family == "nonparametric"

    def test_friedman_rule_min_groups(self):
        """Test Friedman rule attributes."""
        # Arrange
        rule = TEST_RULES["friedman"]
        # Act
        # Assert
        assert rule.min_groups == 3

    def test_friedman_rule_max_groups(self):
        """Test Friedman rule attributes."""
        # Arrange
        rule = TEST_RULES["friedman"]
        # Act
        # Assert
        assert rule.max_groups is None

    def test_friedman_rule_supports_paired(self):
        """Test Friedman rule attributes."""
        # Arrange
        rule = TEST_RULES["friedman"]
        # Act
        # Assert
        assert rule.supports_paired is True

    def test_friedman_rule_within_design_allowed(self):
        """Test Friedman rule attributes."""
        # Arrange
        rule = TEST_RULES["friedman"]
        # Act
        # Assert
        assert "within" in rule.design_allowed

    def test_fisher_exact_rule_family_categorical(self):
        """Test Fisher's exact test rule attributes."""
        # Arrange
        rule = TEST_RULES["fisher_exact"]
        # Act
        # Assert
        assert rule.family == "categorical"

    def test_fisher_exact_rule_min_groups(self):
        """Test Fisher's exact test rule attributes."""
        # Arrange
        rule = TEST_RULES["fisher_exact"]
        # Act
        # Assert
        assert rule.min_groups == 2

    def test_fisher_exact_rule_max_groups(self):
        """Test Fisher's exact test rule attributes."""
        # Arrange
        rule = TEST_RULES["fisher_exact"]
        # Act
        # Assert
        assert rule.max_groups == 2

    def test_fisher_exact_rule_binary_outcome_types(self):
        """Test Fisher's exact test rule attributes."""
        # Arrange
        rule = TEST_RULES["fisher_exact"]
        # Act
        # Assert
        assert "binary" in rule.outcome_types

    def test_fisher_exact_rule_needs_normality(self):
        """Test Fisher's exact test rule attributes."""
        # Arrange
        rule = TEST_RULES["fisher_exact"]
        # Act
        # Assert
        assert rule.needs_normality is False

    def test_dunnett_rule_family_posthoc(self):
        """Test Dunnett's test rule attributes."""
        # Arrange
        rule = TEST_RULES["dunnett"]
        # Act
        # Assert
        assert rule.family == "posthoc"

    def test_dunnett_rule_min_groups(self):
        """Test Dunnett's test rule attributes."""
        # Arrange
        rule = TEST_RULES["dunnett"]
        # Act
        # Assert
        assert rule.min_groups == 3

    def test_dunnett_rule_requires_control_group(self):
        """Test Dunnett's test rule attributes."""
        # Arrange
        rule = TEST_RULES["dunnett"]
        # Act
        # Assert
        assert rule.requires_control_group is True

    def test_dunnett_rule_needs_normality(self):
        """Test Dunnett's test rule attributes."""
        # Arrange
        rule = TEST_RULES["dunnett"]
        # Act
        # Assert
        assert rule.needs_normality is True

    def test_dunnett_rule_needs_equal_variance(self):
        """Test Dunnett's test rule attributes."""
        # Arrange
        rule = TEST_RULES["dunnett"]
        # Act
        # Assert
        assert rule.needs_equal_variance is True

    def test_games_howell_rule_family_posthoc(self):
        """Test Games-Howell rule attributes."""
        # Arrange
        rule = TEST_RULES["games_howell"]
        # Act
        # Assert
        assert rule.family == "posthoc"

    def test_games_howell_rule_min_groups(self):
        """Test Games-Howell rule attributes."""
        # Arrange
        rule = TEST_RULES["games_howell"]
        # Act
        # Assert
        assert rule.min_groups == 3

    def test_games_howell_rule_requires_control_group(self):
        """Test Games-Howell rule attributes."""
        # Arrange
        rule = TEST_RULES["games_howell"]
        # Act
        # Assert
        assert rule.requires_control_group is False

    def test_games_howell_rule_needs_normality(self):
        """Test Games-Howell rule attributes."""
        # Arrange
        rule = TEST_RULES["games_howell"]
        # Act
        # Assert
        assert rule.needs_normality is True

    def test_games_howell_rule_needs_equal_variance(self):
        """Test Games-Howell rule attributes."""
        # Arrange
        rule = TEST_RULES["games_howell"]
        # Act
        # Assert
        assert rule.needs_equal_variance is False


class TestPriorityOrdering:
    """Tests for priority ordering across tests."""

    def test_all_priorities_are_integers(self):
        """Test that all priorities are integers."""
        # Arrange
        # Act
        # Assert
        for name, rule in TEST_RULES.items():
            assert isinstance(rule.priority, int), f"{name} priority should be int"

    def test_priorities_are_reasonable(self):
        """Test that priorities are in a reasonable range."""
        # Arrange
        # Act
        # Assert
        for name, rule in TEST_RULES.items():
            assert 0 <= rule.priority <= 150, (
                f"{name} priority {rule.priority} out of range"
            )

    def test_brunner_munzel_highest_priority_for_two_groups(self):
        """Test that Brunner-Munzel has highest priority among 2-group tests."""
        # Arrange
        two_group_tests = {
            name: rule
            for name, rule in TEST_RULES.items()
            if rule.min_groups == 2 and rule.max_groups == 2
        }
        priorities = {name: rule.priority for name, rule in two_group_tests.items()}
        # Act
        max_priority_test = max(priorities, key=priorities.get)
        # Assert
        assert max_priority_test == "brunner_munzel"

    def test_brunner_munzel_has_highest_priority_brunner_munzel_priority_ttest_ind_priority(
        self,
    ):
        """Test that Brunner-Munzel has the highest priority among 2-group tests.

        Brunner-Munzel is the recommended default due to its robustness
        (no normality or equal variance assumptions). Other tests like
        Mann-Whitney and t-test have lower priorities.
        """
        # Arrange
        brunner_munzel_priority = TEST_RULES["brunner_munzel"].priority
        mannwhitneyu_priority = TEST_RULES["mannwhitneyu"].priority
        ttest_ind_priority = TEST_RULES["ttest_ind"].priority
        # Act
        # Assert
        assert brunner_munzel_priority > ttest_ind_priority

    def test_brunner_munzel_has_highest_priority_brunner_munzel_priority_mannwhitneyu_priority(
        self,
    ):
        """Test that Brunner-Munzel has the highest priority among 2-group tests.

        Brunner-Munzel is the recommended default due to its robustness
        (no normality or equal variance assumptions). Other tests like
        Mann-Whitney and t-test have lower priorities.
        """
        # Arrange
        brunner_munzel_priority = TEST_RULES["brunner_munzel"].priority
        mannwhitneyu_priority = TEST_RULES["mannwhitneyu"].priority
        ttest_ind_priority = TEST_RULES["ttest_ind"].priority
        # Act
        # Assert
        assert brunner_munzel_priority > mannwhitneyu_priority

    def test_brunner_munzel_has_highest_priority_ttest_ind_priority_mannwhitneyu_priority(
        self,
    ):
        """Test that Brunner-Munzel has the highest priority among 2-group tests.

        Brunner-Munzel is the recommended default due to its robustness
        (no normality or equal variance assumptions). Other tests like
        Mann-Whitney and t-test have lower priorities.
        """
        # Arrange
        brunner_munzel_priority = TEST_RULES["brunner_munzel"].priority
        mannwhitneyu_priority = TEST_RULES["mannwhitneyu"].priority
        ttest_ind_priority = TEST_RULES["ttest_ind"].priority
        # Act
        # Assert
        assert ttest_ind_priority > mannwhitneyu_priority


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_rules.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_rules.py
#
# """
# Test Rules - Applicability rules for statistical tests.
#
# This module defines TestRule dataclass and the TEST_RULES registry that
# maps test names to their applicability conditions. Used by check_applicable()
# to determine which tests can be applied to a given StatContext.
#
# The priority field is used for test recommendation - higher priority tests
# are recommended first when multiple tests are applicable.
# """
#
# from __future__ import annotations
#
# from dataclasses import dataclass, field
# from typing import Dict, Literal, Optional, Set
#
# # =============================================================================
# # Type Aliases
# # =============================================================================
#
# TestFamily = Literal[
#     "parametric",
#     "nonparametric",
#     "categorical",
#     "correlation",
#     "normality",
#     "effect_size",
#     "posthoc",
#     "other",
# ]
#
#
# # =============================================================================
# # TestRule
# # =============================================================================
#
#
# @dataclass
# class TestRule:
#     """
#     Applicability rule for a specific statistical test.
#
#     Each TestRule defines the conditions under which a test is applicable.
#     The check_applicable() function uses these rules to filter tests
#     for a given StatContext.
#
#     Parameters
#     ----------
#     name : str
#         Internal name of the test (e.g., "ttest_ind", "brunner_munzel").
#     family : TestFamily
#         High-level family of the test:
#         - "parametric": t-test, ANOVA, etc.
#         - "nonparametric": Mann-Whitney, Kruskal-Wallis, etc.
#         - "categorical": Chi-square, Fisher's exact, etc.
#         - "correlation": Pearson, Spearman, etc.
#         - "normality": Shapiro-Wilk, etc.
#         - "effect_size": Cohen's d, eta-squared, etc.
#         - "posthoc": Tukey, Dunnett, etc.
#         - "other": Other tests (Levene, etc.)
#     min_groups : int
#         Minimum required number of groups.
#     max_groups : int or None
#         Maximum allowed number of groups. None means no upper bound.
#     outcome_types : set of str
#         Allowed outcome types for this test.
#     supports_paired : bool
#         Whether the test supports paired/repeated measures.
#     supports_unpaired : bool
#         Whether the test supports independent groups.
#     design_allowed : set of str
#         Allowed designs, e.g., {"between", "within"}.
#     requires_control_group : bool
#         Whether a dedicated control group is required (e.g., Dunnett).
#     min_n_total : int or None
#         Minimum total sample size. None means no constraint.
#     min_n_per_group : int or None
#         Minimum sample size per group.
#     needs_normality : bool
#         Whether test assumes normality (check normality_ok).
#     needs_equal_variance : bool
#         Whether test assumes equal variances (check variance_homogeneity_ok).
#     min_factors : int or None
#         Minimum number of factors.
#     max_factors : int or None
#         Maximum number of factors.
#     priority : int
#         Priority score for recommendation. Higher = more recommended.
#         Brunner-Munzel has priority 110 as the recommended default for 2 groups.
#     description : str
#         Human-readable description for tooltips.
#
#     Examples
#     --------
#     >>> rule = TestRule(
#     ...     name="ttest_ind",
#     ...     family="parametric",
#     ...     min_groups=2,
#     ...     max_groups=2,
#     ...     outcome_types={"continuous"},
#     ...     supports_paired=False,
#     ...     supports_unpaired=True,
#     ...     design_allowed={"between"},
#     ...     requires_control_group=False,
#     ...     min_n_total=4,
#     ...     min_n_per_group=2,
#     ...     needs_normality=True,
#     ...     needs_equal_variance=False,
#     ...     min_factors=1,
#     ...     max_factors=1,
#     ...     priority=90,
#     ...     description="Independent samples t-test (Welch)"
#     ... )
#     """
#
#     name: str
#     family: TestFamily
#     min_groups: int
#     max_groups: Optional[int]
#     outcome_types: Set[str]
#     supports_paired: bool
#     supports_unpaired: bool
#     design_allowed: Set[str]
#     requires_control_group: bool
#     min_n_total: Optional[int]
#     min_n_per_group: Optional[int]
#     needs_normality: bool
#     needs_equal_variance: bool
#     min_factors: Optional[int]
#     max_factors: Optional[int]
#     priority: int = 0
#     description: str = ""
#
#
# # =============================================================================
# # TEST_RULES Registry
# # =============================================================================
#
# TEST_RULES: Dict[str, TestRule] = {
#     # =========================================================================
#     # Parametric Tests - Mean Comparisons
#     # =========================================================================
#
#     # Independent 2-sample t-test (Welch)
#     "ttest_ind": TestRule(
#         name="ttest_ind",
#         family="parametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,  # Welch doesn't require equal variance
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Independent samples t-test (Welch)",
#     ),
#
#     # Paired t-test
#     "ttest_rel": TestRule(
#         name="ttest_rel",
#         family="parametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=95,
#         description="Paired samples t-test",
#     ),
#
#     # One-way ANOVA (between)
#     "anova_oneway": TestRule(
#         name="anova_oneway",
#         family="parametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=80,
#         description="One-way ANOVA (between subjects)",
#     ),
#
#     # Repeated-measures one-way ANOVA
#     "anova_rm_oneway": TestRule(
#         name="anova_rm_oneway",
#         family="parametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=85,
#         description="Repeated-measures one-way ANOVA",
#     ),
#
#     # Welch ANOVA (unequal variances)
#     "welch_anova": TestRule(
#         name="welch_anova",
#         family="parametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,  # Welch doesn't require equal variance
#         min_factors=1,
#         max_factors=1,
#         priority=82,
#         description="Welch's ANOVA (heterogeneous variances)",
#     ),
#
#     # Two-way ANOVA (between)
#     "anova_twoway": TestRule(
#         name="anova_twoway",
#         family="parametric",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=8,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=2,
#         max_factors=2,
#         priority=78,
#         description="Two-way ANOVA (between subjects)",
#     ),
#
#     # Two-way ANOVA (mixed)
#     "anova_twoway_mixed": TestRule(
#         name="anova_twoway_mixed",
#         family="parametric",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"mixed", "within"},
#         requires_control_group=False,
#         min_n_total=8,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=2,
#         max_factors=2,
#         priority=80,
#         description="Two-way mixed-design ANOVA",
#     ),
#
#     # =========================================================================
#     # Nonparametric Tests - Rank Comparisons
#     # =========================================================================
#
#     # Brunner-Munzel test (RECOMMENDED DEFAULT for 2 groups)
#     "brunner_munzel": TestRule(
#         name="brunner_munzel",
#         family="nonparametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=3,
#         needs_normality=False,
#         needs_equal_variance=False,  # Most robust - no assumptions
#         min_factors=1,
#         max_factors=1,
#         priority=110,  # HIGHEST PRIORITY - recommended default
#         description="Brunner-Munzel test (most robust, recommended)",
#     ),
#
#     # Mann-Whitney U test
#     "mannwhitneyu": TestRule(
#         name="mannwhitneyu",
#         family="nonparametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=85,
#         description="Mann-Whitney U test (rank-sum)",
#     ),
#
#     # Wilcoxon signed-rank test (paired)
#     "wilcoxon": TestRule(
#         name="wilcoxon",
#         family="nonparametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Wilcoxon signed-rank test (paired)",
#     ),
#
#     # Kruskal-Wallis (3+ groups, between)
#     "kruskal": TestRule(
#         name="kruskal",
#         family="nonparametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=75,
#         description="Kruskal-Wallis H test",
#     ),
#
#     # Friedman test (3+ groups, within)
#     "friedman": TestRule(
#         name="friedman",
#         family="nonparametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=80,
#         description="Friedman test (repeated measures)",
#     ),
#
#     # =========================================================================
#     # Categorical Tests
#     # =========================================================================
#
#     # Chi-square test of independence
#     "chi2_independence": TestRule(
#         name="chi2_independence",
#         family="categorical",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"binary", "categorical"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=10,
#         min_n_per_group=None,  # Uses expected counts
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=None,
#         priority=80,
#         description="Chi-square test of independence",
#     ),
#
#     # Fisher's exact test (2x2)
#     "fisher_exact": TestRule(
#         name="fisher_exact",
#         family="categorical",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=1,
#         min_n_per_group=1,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Fisher's exact test (2x2)",
#     ),
#
#     # McNemar's test (paired binary)
#     "mcnemar": TestRule(
#         name="mcnemar",
#         family="categorical",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=85,
#         description="McNemar's test (paired binary)",
#     ),
#
#     # =========================================================================
#     # Correlation Tests
#     # =========================================================================
#
#     # Pearson correlation
#     "pearsonr": TestRule(
#         name="pearsonr",
#         family="correlation",
#         min_groups=1,
#         max_groups=1,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=True,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=80,
#         description="Pearson correlation coefficient",
#     ),
#
#     # Spearman correlation
#     "spearmanr": TestRule(
#         name="spearmanr",
#         family="correlation",
#         min_groups=1,
#         max_groups=1,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=85,
#         description="Spearman rank correlation",
#     ),
#
#     # =========================================================================
#     # Normality Tests
#     # =========================================================================
#
#     # Shapiro-Wilk test
#     "shapiro": TestRule(
#         name="shapiro",
#         family="normality",
#         min_groups=1,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=60,
#         description="Shapiro-Wilk normality test",
#     ),
#
#     # Levene's test for homogeneity of variance
#     "levene": TestRule(
#         name="levene",
#         family="other",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=None,
#         priority=70,
#         description="Levene's test for homogeneity of variance",
#     ),
#
#     # =========================================================================
#     # Post-hoc Tests
#     # =========================================================================
#
#     # Tukey HSD
#     "tukey_hsd": TestRule(
#         name="tukey_hsd",
#         family="posthoc",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=88,
#         description="Tukey HSD post-hoc test",
#     ),
#
#     # Dunnett (control vs treatments)
#     "dunnett": TestRule(
#         name="dunnett",
#         family="posthoc",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=True,  # REQUIRES control group
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=86,
#         description="Dunnett's test (control vs treatments)",
#     ),
#
#     # Games-Howell (unequal variances)
#     "games_howell": TestRule(
#         name="games_howell",
#         family="posthoc",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,  # Does NOT require equal variance
#         min_factors=1,
#         max_factors=1,
#         priority=89,
#         description="Games-Howell post-hoc (unequal variances)",
#     ),
#
#     # =========================================================================
#     # Effect Size Measures
#     # =========================================================================
#
#     # Cohen's d (independent)
#     "cohens_d_ind": TestRule(
#         name="cohens_d_ind",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Cohen's d (independent samples)",
#     ),
#
#     # Cohen's d (paired)
#     "cohens_d_paired": TestRule(
#         name="cohens_d_paired",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=92,
#         description="Cohen's d (paired samples)",
#     ),
#
#     # Hedges' g
#     "hedges_g": TestRule(
#         name="hedges_g",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=88,
#         description="Hedges' g (bias-corrected effect size)",
#     ),
#
#     # Cliff's delta
#     "cliffs_delta": TestRule(
#         name="cliffs_delta",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=86,
#         description="Cliff's delta (nonparametric effect size)",
#     ),
#
#     # Eta-squared
#     "eta_squared": TestRule(
#         name="eta_squared",
#         family="effect_size",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=80,
#         description="Eta-squared (variance explained)",
#     ),
#
#     # Partial eta-squared
#     "partial_eta_squared": TestRule(
#         name="partial_eta_squared",
#         family="effect_size",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=None,
#         priority=85,
#         description="Partial eta-squared (multi-factor designs)",
#     ),
#
#     # Effect size r (for correlations)
#     "effect_size_r": TestRule(
#         name="effect_size_r",
#         family="effect_size",
#         min_groups=1,
#         max_groups=1,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=80,
#         description="Effect size r (correlation)",
#     ),
#
#     # Odds ratio
#     "odds_ratio": TestRule(
#         name="odds_ratio",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=1,
#         min_n_per_group=1,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=88,
#         description="Odds ratio (2x2 table)",
#     ),
#
#     # Risk ratio
#     "risk_ratio": TestRule(
#         name="risk_ratio",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=1,
#         min_n_per_group=1,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=86,
#         description="Risk ratio (relative risk)",
#     ),
#
#     # Probability of superiority P(X>Y)
#     "prob_superiority": TestRule(
#         name="prob_superiority",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=84,
#         description="Probability of superiority P(X>Y)",
#     ),
# }
#
#
# # =============================================================================
# # Utility Functions
# # =============================================================================
#
#
# def get_test_rule(name: str) -> Optional[TestRule]:
#     """
#     Get a TestRule by name.
#
#     Parameters
#     ----------
#     name : str
#         Test name (e.g., "ttest_ind", "brunner_munzel").
#
#     Returns
#     -------
#     TestRule or None
#         The TestRule if found, else None.
#     """
#     return TEST_RULES.get(name)
#
#
# def list_tests_by_family(family: TestFamily) -> Dict[str, TestRule]:
#     """
#     Get all tests in a specific family.
#
#     Parameters
#     ----------
#     family : TestFamily
#         Test family to filter by.
#
#     Returns
#     -------
#     dict
#         Dictionary of test name -> TestRule for the family.
#     """
#     return {
#         name: rule for name, rule in TEST_RULES.items()
#         if rule.family == family
#     }
#
#
# # =============================================================================
# # Public API
# # =============================================================================
#
# __all__ = [
#     "TestRule",
#     "TestFamily",
#     "TEST_RULES",
#     "get_test_rule",
#     "list_tests_by_family",
# ]
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_rules.py
# --------------------------------------------------------------------------------
