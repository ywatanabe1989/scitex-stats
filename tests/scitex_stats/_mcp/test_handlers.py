"""Tests for ``scitex_stats._mcp.handlers`` re-exports + key handlers.

Async handlers are driven via ``asyncio.run`` so the suite doesn't
depend on pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import math

from scitex_stats._mcp import handlers as h


def _arun(coro):
    return asyncio.run(coro)


# ----- module surface ------------------------------------------------------ #


def test_handlers_module_exports_all_expected_names():
    expected = {
        "recommend_tests_handler",
        "run_test_handler",
        "format_results_handler",
        "power_analysis_handler",
        "correct_pvalues_handler",
        "describe_handler",
        "effect_size_handler",
        "normality_test_handler",
        "posthoc_test_handler",
        "p_to_stars_handler",
    }
    assert expected.issubset(set(h.__all__))
    for name in expected:
        assert hasattr(h, name), f"missing: {name}"


# ----- p_to_stars_handler -------------------------------------------------- #


def test_p_to_stars_three_stars_for_tiny_p():
    out = _arun(h.p_to_stars_handler(p_value=0.0001))
    assert out["success"] is True
    assert out["stars"] == "***"


def test_p_to_stars_two_stars_for_p_under_0_01():
    out = _arun(h.p_to_stars_handler(p_value=0.005))
    assert out["stars"] == "**"


def test_p_to_stars_one_star_for_p_under_0_05():
    out = _arun(h.p_to_stars_handler(p_value=0.04))
    assert out["stars"] == "*"


def test_p_to_stars_ns_for_p_over_0_05():
    out = _arun(h.p_to_stars_handler(p_value=0.20))
    assert out["stars"] == "ns"


def test_p_to_stars_custom_thresholds():
    """Caller-supplied thresholds should override defaults."""
    out = _arun(h.p_to_stars_handler(p_value=0.05, thresholds=[0.0001, 0.001, 0.01]))
    # 0.05 ≥ 0.01 → not significant under tightened thresholds.
    assert out["stars"] == "ns"


# ----- describe_handler ---------------------------------------------------- #


def test_describe_basic_stats():
    """1..10 → mean=5.5, n=10, range=9."""
    out = _arun(h.describe_handler(data=[float(i) for i in range(1, 11)]))
    assert out["success"] is True
    assert out["n"] == 10
    assert math.isclose(out["mean"], 5.5)
    assert math.isclose(out["range"], 9.0)


def test_describe_filters_nan():
    out = _arun(h.describe_handler(data=[1.0, 2.0, float("nan"), 3.0]))
    assert out["n"] == 3


def test_describe_default_percentiles_present():
    out = _arun(h.describe_handler(data=list(range(100))))
    assert "percentiles" in out
    assert {"25", "50", "75"}.issubset(set(out["percentiles"].keys()))


def test_describe_custom_percentiles():
    out = _arun(h.describe_handler(data=list(range(100)), percentiles=[10, 90]))
    assert {"10", "90"}.issubset(set(out["percentiles"].keys()))


def test_describe_handles_single_point():
    """n=1: std/var/sem default to 0.0 (avoid division by n-1)."""
    out = _arun(h.describe_handler(data=[42.0]))
    assert out["n"] == 1
    assert out["std"] == 0.0
    assert out["mean"] == 42.0


def test_describe_iqr_correct():
    """For 1..101, IQR (Q3-Q1) ≈ 50."""
    out = _arun(h.describe_handler(data=list(range(1, 102))))
    assert math.isclose(out["iqr"], 50.0, abs_tol=1.0)


def test_describe_skewness_present_when_scipy_available():
    out = _arun(h.describe_handler(data=list(range(1, 50))))
    # scipy is a hard dep of scitex-stats, so skewness should be present.
    assert "skewness" in out
    assert isinstance(out["skewness"], float)


# ----- posthoc_test_handler ------------------------------------------------ #

import numpy as np

_RNG = np.random.default_rng(0)
_G1 = _RNG.normal(0.0, 1.0, 30).tolist()
_G2 = _RNG.normal(0.5, 1.0, 30).tolist()
_G3 = _RNG.normal(1.0, 1.0, 30).tolist()


def test_posthoc_tukey_returns_three_comparisons_for_three_groups():
    out = _arun(
        h.posthoc_test_handler(
            groups=[_G1, _G2, _G3],
            group_names=["A", "B", "C"],
            method="tukey",
        )
    )
    assert out["success"] is True
    assert out["method"] == "tukey"
    assert out["n_groups"] == 3
    assert out["n_comparisons"] == 3
    pair_keys = {(c.get("group1"), c.get("group2")) for c in out["comparisons"]}
    assert ("A", "B") in pair_keys or ("B", "A") in pair_keys


def test_posthoc_games_howell_handles_unequal_variances():
    g_hi_var = _RNG.normal(0.0, 3.0, 30).tolist()
    out = _arun(
        h.posthoc_test_handler(
            groups=[_G1, g_hi_var, _G3],
            group_names=["A", "B", "C"],
            method="games_howell",
        )
    )
    assert out["success"] is True
    assert out["method"] == "games_howell"
    assert out["n_comparisons"] == 3


def test_posthoc_dunnett_compares_each_to_control():
    out = _arun(
        h.posthoc_test_handler(
            groups=[_G1, _G2, _G3],
            group_names=["ctrl", "B", "C"],
            method="dunnett",
            control_group=0,
        )
    )
    assert out["success"] is True
    assert out["method"] == "dunnett"
    # Dunnett: control vs each other => k-1 comparisons.
    assert out["n_comparisons"] == 2


def test_posthoc_dunn_runs_for_nonparametric_use():
    out = _arun(
        h.posthoc_test_handler(
            groups=[_G1, _G2, _G3],
            group_names=["A", "B", "C"],
            method="dunn",
        )
    )
    assert out["success"] is True
    assert out["method"] == "dunn"
    assert out["n_comparisons"] == 3


def test_posthoc_rejects_unknown_method():
    out = _arun(
        h.posthoc_test_handler(
            groups=[_G1, _G2], method="not_a_real_method"
        )
    )
    assert out["success"] is False
    assert "not_a_real_method" in out["error"]


def test_posthoc_default_group_names_are_indexed_one_based():
    out = _arun(h.posthoc_test_handler(groups=[_G1, _G2, _G3]))
    assert out["success"] is True
    names = {c.get("group1") for c in out["comparisons"]} | {
        c.get("group2") for c in out["comparisons"]
    }
    assert names == {"Group_1", "Group_2", "Group_3"}


# ----- power_analysis_handler --------------------------------------------- #


def test_power_ttest_calculates_power_when_n_and_effect_given():
    out = _arun(
        h.power_analysis_handler(
            test_type="ttest", effect_size=0.5, n=30, alpha=0.05
        )
    )
    assert out["success"] is True
    assert out["mode"] == "power_calculation"
    assert 0.0 < out["power"] < 1.0
    assert out["n1"] == 30
    assert out["effect_size"] == 0.5


def test_power_ttest_calculates_sample_size_when_only_effect_given():
    out = _arun(
        h.power_analysis_handler(
            test_type="ttest", effect_size=0.5, power=0.8, alpha=0.05
        )
    )
    assert out["success"] is True
    assert out["mode"] == "sample_size_calculation"
    assert out["required_n1"] > 0
    assert out["required_n2"] > 0
    assert out["total_n"] == out["required_n1"] + out["required_n2"]


def test_power_ttest_errors_when_neither_n_nor_effect_given():
    out = _arun(h.power_analysis_handler(test_type="ttest"))
    assert out["success"] is False
    assert "n or effect_size" in out["error"]


def test_power_anova_returns_a_dict():
    out = _arun(
        h.power_analysis_handler(
            test_type="anova",
            effect_size=0.3,
            power=0.8,
            alpha=0.05,
            n_groups=3,
        )
    )
    assert out["success"] is True
    assert out["test_type"] == "anova"


def test_power_correlation_returns_a_dict():
    out = _arun(
        h.power_analysis_handler(
            test_type="correlation",
            effect_size=0.3,
            power=0.8,
            alpha=0.05,
        )
    )
    assert out["success"] is True
    assert out["test_type"] == "correlation"


def test_power_chi2_returns_a_dict():
    out = _arun(
        h.power_analysis_handler(
            test_type="chi2",
            effect_size=0.3,
            power=0.8,
            alpha=0.05,
            n_groups=4,
        )
    )
    assert out["success"] is True
    assert out["test_type"] == "chi2"


def test_power_rejects_unknown_test_type():
    out = _arun(h.power_analysis_handler(test_type="not_a_real_test"))
    assert out["success"] is False
    assert "not_a_real_test" in out["error"]
