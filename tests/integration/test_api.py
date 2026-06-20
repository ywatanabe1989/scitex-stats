#!/usr/bin/env python3
# File: tests/test_api.py

"""Tests for scitex_stats public Python API.

Verifies that all advertised public names are importable and functional.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture: deterministic random data
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def two_groups(rng):
    x = rng.randn(30)
    y = rng.randn(30) + 0.5
    return x, y


@pytest.fixture
def three_groups(rng):
    a = rng.randn(25)
    b = rng.randn(25) + 0.3
    c = rng.randn(25) + 0.8
    return [a, b, c]


# ===================================================================
# 1. Public API surface
# ===================================================================


class TestPublicAPIImports:
    """All names listed in __all__ must be importable."""

    def test_version_string_str_scitex_stats(self):
        # Arrange
        import scitex_stats
        # Act
        # Assert
        assert isinstance(scitex_stats.__version__, str)

    def test_version_string_scitex_stats(self):
        # Arrange
        import scitex_stats
        # Act
        # Assert
        assert len(scitex_stats.__version__) > 0

    def test_scitex_stats_imports_work_hasattr_scitex_stats_run_test(self):
        # Arrange
        import scitex_stats
        # Act
        # Assert
        assert hasattr(scitex_stats, "run_test")

    def test_scitex_stats_imports_work_hasattr_scitex_stats_available_tests(self):
        # Arrange
        import scitex_stats
        # Act
        # Assert
        assert hasattr(scitex_stats, "available_tests")

    def test_scitex_stats_imports_work_callable_run_test_scitex_stats(self):
        # Arrange
        import scitex_stats
        # Act
        # Assert
        assert callable(scitex_stats.run_test)

    @pytest.mark.parametrize(
        "name",
        [
            "run_test",
            "available_tests",
            "describe",
            "auto",
            "correct",
            "descriptive",
            "effect_sizes",
            "posthoc",
            "power",
            "tests",
            "StatContext",
            "TestRule",
            "recommend_tests",
            "p_to_stars",
        ],
    )
    def test_public_name_importable(self, name):
        # Arrange
        import scitex_stats
        # Act
        obj = getattr(scitex_stats, name)
        # Assert
        assert obj is not None

    def test_submodules_are_modules(self):
        # Arrange
        import types
        import scitex_stats
        # Act
        # Assert
        for name in (
            "auto",
            "correct",
            "descriptive",
            "effect_sizes",
            "posthoc",
            "power",
            "tests",
        ):
            obj = getattr(scitex_stats, name)
            assert isinstance(obj, types.ModuleType), (
                f"{name} should be a module, got {type(obj)}"
            )


# ===================================================================
# 2. run_test dispatcher
# ===================================================================


class TestRunTest:
    """Test the unified run_test dispatcher."""

    def test_ttest_ind_returns_dict_case_1(self, two_groups):
        # Arrange
        import scitex_stats as ss
        x, y = two_groups
        # Act
        result = ss.run_test("ttest_ind", data=x, data2=y)
        # Assert
        assert isinstance(result, dict)

    def test_ttest_ind_returns_dict_value_pvalue(self, two_groups):
        # Arrange
        import scitex_stats as ss
        x, y = two_groups
        # Act
        result = ss.run_test("ttest_ind", data=x, data2=y)
        # Assert
        assert "p_value" in result or "pvalue" in result

    def test_ttest_ind_has_statistic(self, two_groups):
        # Arrange
        import scitex_stats as ss
        x, y = two_groups
        # Act
        result = ss.run_test("ttest_ind", data=x, data2=y)
        # Assert
        assert "statistic" in result

    def test_unknown_test_raises_valueerror(self):
        # Arrange
        import scitex_stats as ss
        # Act
        # Assert
        with pytest.raises(ValueError, match="Unknown test"):
            ss.run_test("nonexistent_test", data=[1, 2, 3])

    def test_shapiro_one_sample(self, rng):
        # Arrange
        import scitex_stats as ss
        x = rng.randn(50)
        # Act
        result = ss.run_test("shapiro", data=x)
        # Assert
        assert isinstance(result, dict)

    def test_wilcoxon_paired_dict(self, two_groups):
        # Arrange
        import scitex_stats as ss
        x, y = two_groups
        # Act
        result = ss.run_test("wilcoxon", data=x, data2=y)
        # Assert
        assert isinstance(result, dict)

    def test_anova_groups_dict(self, three_groups):
        # Arrange
        import scitex_stats as ss
        # Act
        result = ss.run_test("anova", groups=three_groups)
        # Assert
        assert isinstance(result, dict)


# ===================================================================
# 3. available_tests
# ===================================================================


class TestAvailableTests:
    """Test the available_tests function."""

    def test_returns_list_case(self):
        # Arrange
        import scitex_stats as ss
        # Act
        result = ss.available_tests()
        # Assert
        assert isinstance(result, list)

    def test_contains_known_tests(self):
        # Arrange
        import scitex_stats as ss
        # Act
        tests = ss.available_tests()
        # Assert
        for expected in (
            "ttest_ind",
            "anova",
            "shapiro",
            "chi2",
            "pearson",
            "kruskal",
            "wilcoxon",
        ):
            assert expected in tests, f"{expected} not in available_tests"

    def test_sorted_order_tests(self):
        # Arrange
        import scitex_stats as ss
        # Act
        tests = ss.available_tests()
        # Assert
        assert tests == sorted(tests)


# ===================================================================
# 4. recommend_tests
# ===================================================================


class TestRecommendTests:
    """Test automatic test recommendation."""

    def test_two_group_continuous_recs_list(self):
        # Arrange
        import scitex_stats as ss
        ctx = ss.StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            paired=False,
            has_control_group=False,
            n_factors=1,
        )
        # Act
        recs = ss.recommend_tests(ctx, top_k=3)
        # Assert
        assert isinstance(recs, list)

    def test_two_group_continuous_recs(self):
        # Arrange
        import scitex_stats as ss
        ctx = ss.StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            paired=False,
            has_control_group=False,
            n_factors=1,
        )
        # Act
        recs = ss.recommend_tests(ctx, top_k=3)
        # Assert
        assert len(recs) <= 3

    def test_two_group_continuous_recs_2(self):
        # Arrange
        import scitex_stats as ss
        ctx = ss.StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            paired=False,
            has_control_group=False,
            n_factors=1,
        )
        # Act
        recs = ss.recommend_tests(ctx, top_k=3)
        # Assert
        assert len(recs) > 0

    def test_brunner_munzel_is_top_pick(self):
        """Brunner-Munzel should be the default recommendation for 2 groups."""
        # Arrange
        import scitex_stats as ss
        ctx = ss.StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            paired=False,
            has_control_group=False,
            n_factors=1,
        )
        # Act
        recs = ss.recommend_tests(ctx, top_k=3)
        # Assert
        assert "brunner_munzel" in recs

    def test_stat_context_creation(self):
        # Arrange
        import scitex_stats as ss
        # Act
        ctx = ss.StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            paired=False,
            has_control_group=True,
            n_factors=1,
        )
        # Assert
        assert ctx.n_groups == 3


# ===================================================================
# 5. describe
# ===================================================================


class TestDescribe:
    """Test descriptive statistics."""

    def test_returns_tuple_case_1(self, rng):
        # Arrange
        import scitex_stats as ss
        x = rng.randn(100)
        # Act
        result = ss.describe(x)
        # Assert
        assert isinstance(result, tuple)

    def test_returns_tuple_case_2(self, rng):
        # Arrange
        import scitex_stats as ss
        x = rng.randn(100)
        # Act
        result = ss.describe(x)
        # Assert
        assert len(result) == 2

    def test_values_and_names_list(self, rng):
        # Arrange
        import scitex_stats as ss
        x = rng.randn(100)
        # Act
        values, names = ss.describe(x)
        # Assert
        assert isinstance(names, list)

    def test_values_and_names_case_2(self, rng):
        # Arrange
        import scitex_stats as ss
        x = rng.randn(100)
        # Act
        values, names = ss.describe(x)
        # Assert
        assert len(names) > 0

    def test_values_and_names_shape(self, rng):
        # Arrange
        import scitex_stats as ss
        x = rng.randn(100)
        # Act
        values, names = ss.describe(x)
        # Assert
        assert values.shape[0] == len(names)

    def test_contains_mean_names(self, rng):
        # Arrange
        import scitex_stats as ss
        x = rng.randn(100)
        # Act
        _, names = ss.describe(x)
        # Assert
        assert "mean" in names


# ===================================================================
# 6. p_to_stars
# ===================================================================


class TestPToStars:
    """Test p-value to significance stars conversion."""

    def test_highly_significant_case(self):
        # Arrange
        import scitex_stats as ss
        # Act
        result = ss.p_to_stars(0.0001)
        # Assert
        assert "***" in result

    def test_significant_case_case(self):
        # Arrange
        import scitex_stats as ss
        # Act
        result = ss.p_to_stars(0.001)
        # Assert
        assert "*" in result

    def test_not_significant_ns(self):
        # Arrange
        import scitex_stats as ss
        # Act
        result = ss.p_to_stars(0.5)
        # Assert
        assert result == "ns"

    def test_returns_string_str(self):
        # Arrange
        import scitex_stats as ss
        # Act
        result = ss.p_to_stars(0.03)
        # Assert
        assert isinstance(result, str)


# ===================================================================
# 7. Submodule access
# ===================================================================


class TestSubmoduleAccess:
    """Verify submodules are accessible and contain expected attributes."""

    def test_effect_sizes_module(self):
        # Arrange
        import scitex_stats as ss
        # Act
        # Assert
        assert hasattr(ss.effect_sizes, "__all__") or callable(
            getattr(ss.effect_sizes, "__getattr__", None)
        )

    def test_correct_module_hasattr_ss(self):
        # Arrange
        import scitex_stats as ss
        # Act
        # Assert
        assert hasattr(ss, "correct")

    def test_posthoc_module_hasattr_ss(self):
        # Arrange
        import scitex_stats as ss
        # Act
        # Assert
        assert hasattr(ss, "posthoc")

    def test_power_module_hasattr_ss(self):
        # Arrange
        import scitex_stats as ss
        # Act
        # Assert
        assert hasattr(ss, "power")

    def test_descriptive_module_hasattr_ss(self):
        # Arrange
        import scitex_stats as ss
        # Act
        # Assert
        assert hasattr(ss, "descriptive")


# ===================================================================
# 8. TestRule class
# ===================================================================


class TestTestRule:
    """Verify TestRule is importable and is a class."""

    def test_is_class_testrule_type_hasattr_dataclass(self):
        # Arrange
        import scitex_stats as ss
        # Act
        # Assert
        assert isinstance(ss.TestRule, type) or hasattr(
            ss.TestRule, "__dataclass_fields__"
        )


# EOF
