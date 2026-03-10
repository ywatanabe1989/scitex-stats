#!/usr/bin/env python3
# File: tests/test_api.py

"""Tests for scitex_stats public Python API.

Verifies that the thin wrapper correctly delegates to scitex.stats
and that all advertised public names are importable and functional.
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

    def test_version_string(self):
        import scitex_stats

        assert isinstance(scitex_stats.__version__, str)
        assert len(scitex_stats.__version__) > 0

    def test_scitex_stats_imports_work(self):
        import scitex_stats

        assert hasattr(scitex_stats, "run_test")
        assert hasattr(scitex_stats, "available_tests")
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
        import scitex_stats

        obj = getattr(scitex_stats, name)
        assert obj is not None

    def test_submodules_are_modules(self):
        import types

        import scitex_stats

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

    def test_ttest_ind_returns_dict(self, two_groups):
        import scitex_stats as ss

        x, y = two_groups
        result = ss.run_test("ttest_ind", data=x, data2=y)
        assert isinstance(result, dict)
        assert "p_value" in result or "pvalue" in result

    def test_ttest_ind_has_statistic(self, two_groups):
        import scitex_stats as ss

        x, y = two_groups
        result = ss.run_test("ttest_ind", data=x, data2=y)
        assert "statistic" in result

    def test_unknown_test_raises_valueerror(self):
        import scitex_stats as ss

        with pytest.raises(ValueError, match="Unknown test"):
            ss.run_test("nonexistent_test", data=[1, 2, 3])

    def test_shapiro_one_sample(self, rng):
        import scitex_stats as ss

        x = rng.randn(50)
        result = ss.run_test("shapiro", data=x)
        assert isinstance(result, dict)

    def test_wilcoxon_paired(self, two_groups):
        import scitex_stats as ss

        x, y = two_groups
        result = ss.run_test("wilcoxon", data=x, data2=y)
        assert isinstance(result, dict)

    def test_anova_groups(self, three_groups):
        import scitex_stats as ss

        result = ss.run_test("anova", groups=three_groups)
        assert isinstance(result, dict)


# ===================================================================
# 3. available_tests
# ===================================================================


class TestAvailableTests:
    """Test the available_tests function."""

    def test_returns_list(self):
        import scitex_stats as ss

        result = ss.available_tests()
        assert isinstance(result, list)

    def test_contains_known_tests(self):
        import scitex_stats as ss

        tests = ss.available_tests()
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

    def test_sorted_order(self):
        import scitex_stats as ss

        tests = ss.available_tests()
        assert tests == sorted(tests)


# ===================================================================
# 4. recommend_tests
# ===================================================================


class TestRecommendTests:
    """Test automatic test recommendation."""

    def test_two_group_continuous(self):
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
        recs = ss.recommend_tests(ctx, top_k=3)
        assert isinstance(recs, list)
        assert len(recs) <= 3
        assert len(recs) > 0

    def test_brunner_munzel_is_top_pick(self):
        """Brunner-Munzel should be the default recommendation for 2 groups."""
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
        recs = ss.recommend_tests(ctx, top_k=3)
        assert "brunner_munzel" in recs

    def test_stat_context_creation(self):
        import scitex_stats as ss

        ctx = ss.StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            paired=False,
            has_control_group=True,
            n_factors=1,
        )
        assert ctx.n_groups == 3


# ===================================================================
# 5. describe
# ===================================================================


class TestDescribe:
    """Test descriptive statistics."""

    def test_returns_tuple(self, rng):
        import scitex_stats as ss

        x = rng.randn(100)
        result = ss.describe(x)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_values_and_names(self, rng):
        import scitex_stats as ss

        x = rng.randn(100)
        values, names = ss.describe(x)
        assert isinstance(names, list)
        assert len(names) > 0
        assert values.shape[0] == len(names)

    def test_contains_mean(self, rng):
        import scitex_stats as ss

        x = rng.randn(100)
        _, names = ss.describe(x)
        assert "mean" in names


# ===================================================================
# 6. p_to_stars
# ===================================================================


class TestPToStars:
    """Test p-value to significance stars conversion."""

    def test_highly_significant(self):
        import scitex_stats as ss

        result = ss.p_to_stars(0.0001)
        assert "***" in result

    def test_significant(self):
        import scitex_stats as ss

        result = ss.p_to_stars(0.001)
        assert "*" in result

    def test_not_significant(self):
        import scitex_stats as ss

        result = ss.p_to_stars(0.5)
        assert result == "ns"

    def test_returns_string(self):
        import scitex_stats as ss

        result = ss.p_to_stars(0.03)
        assert isinstance(result, str)


# ===================================================================
# 7. Submodule access
# ===================================================================


class TestSubmoduleAccess:
    """Verify submodules are accessible and contain expected attributes."""

    def test_effect_sizes_module(self):
        import scitex_stats as ss

        assert hasattr(ss.effect_sizes, "__all__") or callable(
            getattr(ss.effect_sizes, "__getattr__", None)
        )

    def test_correct_module(self):
        import scitex_stats as ss

        assert hasattr(ss, "correct")

    def test_posthoc_module(self):
        import scitex_stats as ss

        assert hasattr(ss, "posthoc")

    def test_power_module(self):
        import scitex_stats as ss

        assert hasattr(ss, "power")

    def test_descriptive_module(self):
        import scitex_stats as ss

        assert hasattr(ss, "descriptive")


# ===================================================================
# 8. TestRule class
# ===================================================================


class TestTestRule:
    """Verify TestRule is importable and is a class."""

    def test_is_class(self):
        import scitex_stats as ss

        assert isinstance(ss.TestRule, type) or hasattr(
            ss.TestRule, "__dataclass_fields__"
        )


# EOF
