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
    out = _arun(h.posthoc_test_handler(groups=[_G1, _G2], method="not_a_real_method"))
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
        h.power_analysis_handler(test_type="ttest", effect_size=0.5, n=30, alpha=0.05)
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


# ----- normality_test_handler -------------------------------------------- #


def test_normality_shapiro_on_normal_data():
    rng_n = np.random.default_rng(0)
    out = _arun(
        h.normality_test_handler(data=rng_n.normal(0, 1, 50).tolist(), method="shapiro")
    )
    assert out["success"] is True
    assert out["test"] == "Shapiro-Wilk"
    assert out["statistic_name"] == "W"
    assert "is_normal" in out


def test_normality_dagostino_on_normal_data():
    rng_n = np.random.default_rng(0)
    out = _arun(
        h.normality_test_handler(
            data=rng_n.normal(0, 1, 50).tolist(), method="dagostino"
        )
    )
    assert out["success"] is True
    assert out["test"] == "D'Agostino-Pearson"


def test_normality_dagostino_rejects_too_few_samples():
    out = _arun(h.normality_test_handler(data=[1.0, 2.0, 3.0, 4.0], method="dagostino"))
    assert out["success"] is True
    assert "error" in out
    assert "8 samples" in out["error"]


def test_normality_anderson_returns_critical_value():
    rng_n = np.random.default_rng(0)
    out = _arun(
        h.normality_test_handler(
            data=rng_n.normal(0, 1, 50).tolist(), method="anderson"
        )
    )
    assert out["success"] is True
    assert out["test"] == "Anderson-Darling"
    assert "critical_value_5pct" in out
    assert "normal" in out


def test_normality_rejects_too_few_overall():
    out = _arun(h.normality_test_handler(data=[1.0, 2.0]))
    assert "error" in out


def test_normality_rejects_unknown_method():
    out = _arun(
        h.normality_test_handler(data=[1.0, 2.0, 3.0, 4.0, 5.0], method="bogus")
    )
    assert out["success"] is False
    assert "bogus" in out["error"]


# ----- effect_size_handler ------------------------------------------------ #

_E_RNG = np.random.default_rng(7)
_E_G1 = _E_RNG.normal(0.0, 1.0, 30).tolist()
_E_G2 = _E_RNG.normal(0.8, 1.0, 30).tolist()


def test_effect_size_cohens_d():
    out = _arun(h.effect_size_handler(group1=_E_G1, group2=_E_G2, measure="cohens_d"))
    assert out["success"] is True
    assert out["measure"] == "Cohen's d"
    assert "value" in out and "interpretation" in out
    assert "ci_lower" in out and "ci_upper" in out
    assert out["ci_lower"] < out["value"] < out["ci_upper"]


def test_effect_size_hedges_g_applies_bias_correction():
    out = _arun(h.effect_size_handler(group1=_E_G1, group2=_E_G2, measure="hedges_g"))
    assert out["success"] is True
    assert out["measure"] == "Hedges' g"


def test_effect_size_glass_delta():
    out = _arun(
        h.effect_size_handler(group1=_E_G1, group2=_E_G2, measure="glass_delta")
    )
    assert out["success"] is True
    assert out["measure"] == "Glass's delta"


def test_effect_size_cliffs_delta_skips_ci():
    out = _arun(
        h.effect_size_handler(group1=_E_G1, group2=_E_G2, measure="cliffs_delta")
    )
    assert out["success"] is True
    assert out["measure"] == "Cliff's delta"
    # CI only added for parametric d-family
    assert "ci_lower" not in out
    assert "ci_upper" not in out


def test_effect_size_rejects_unknown_measure():
    out = _arun(h.effect_size_handler(group1=_E_G1, group2=_E_G2, measure="bogus"))
    assert out["success"] is False
    assert "bogus" in out["error"]


# ----- recommend_tests_handler ------------------------------------------- #


def test_recommend_tests_two_groups_continuous():
    out = _arun(
        h.recommend_tests_handler(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="between",
            paired=False,
            top_k=3,
        )
    )
    assert out["success"] is True
    assert len(out["recommendations"]) <= 3
    assert len(out["recommendations"]) > 0
    first = out["recommendations"][0]
    assert "name" in first and "family" in first and "rationale" in first


def test_recommend_tests_includes_context_echo():
    out = _arun(h.recommend_tests_handler(n_groups=3, design="between"))
    assert out["success"] is True
    assert out["context"]["n_groups"] == 3
    assert out["context"]["design"] == "between"


def test_recommend_tests_default_sample_sizes_use_30():
    # sample_sizes=None should not raise; default-30 per group inside the handler.
    out = _arun(h.recommend_tests_handler(n_groups=2, sample_sizes=None))
    assert out["success"] is True


# ----- format_results_handler -------------------------------------------- #


def test_format_results_apa_returns_formatted_string():
    out = _arun(
        h.format_results_handler(
            test_name="ttest_ind",
            statistic=-3.21,
            p_value=0.0022,
            df=58,
            effect_size=-0.83,
            effect_size_name="d",
            style="apa",
        )
    )
    assert out["success"] is True
    assert out["style"] == "apa"
    assert isinstance(out["formatted"], str) and out["formatted"]
    assert out["stars"]  # at least "*"


def test_format_results_nature_style_works():
    out = _arun(
        h.format_results_handler(
            test_name="ttest_ind",
            statistic=2.0,
            p_value=0.04,
            style="nature",
        )
    )
    assert out["success"] is True
    assert out["style"] == "nature"


def test_format_results_unknown_style_falls_back_to_apa_id():
    out = _arun(
        h.format_results_handler(
            test_name="ttest_ind",
            statistic=2.0,
            p_value=0.04,
            style="not_a_real_style",
        )
    )
    # Handler maps unknown → apa_latex internally; should still succeed.
    assert out["success"] is True


# ----- run_test_handler -------------------------------------------------- #


def test_run_test_ttest_ind_via_handler():
    out = _arun(
        h.run_test_handler(
            test_name="ttest_ind",
            data=[_E_G1, _E_G2],
            alternative="two-sided",
        )
    )
    assert out["success"] is True
    assert out["test_name"] == "ttest_ind"
    assert "statistic" in out and "p_value" in out


def test_run_test_one_sample():
    out = _arun(h.run_test_handler(test_name="ttest_1samp", data=[_E_G1]))
    assert out["success"] is True
    assert out["test_name"] == "ttest_1samp"


def test_run_test_rejects_unknown_test_name():
    out = _arun(h.run_test_handler(test_name="not_a_real_test", data=[_E_G1, _E_G2]))
    assert out["success"] is False
    assert "not_a_real_test" in out["error"]


def test_run_test_rejects_missing_data():
    out = _arun(h.run_test_handler(test_name="ttest_ind"))
    assert out["success"] is False
    assert "data" in out["error"].lower()


# ----- run_test_handler: remaining branches ---------------------------- #


_T_RNG = np.random.default_rng(11)


def test_run_test_multi_group_anova():
    groups = [_T_RNG.normal(0, 1, 25).tolist() for _ in range(3)]
    out = _arun(h.run_test_handler(test_name="anova", data=groups))
    assert out["success"] is True
    assert out["test_name"] == "anova"
    assert "statistic" in out


def test_run_test_multi_group_kruskal():
    groups = [_T_RNG.normal(0, 1, 25).tolist() for _ in range(3)]
    out = _arun(h.run_test_handler(test_name="kruskal", data=groups))
    assert out["success"] is True
    assert out["test_name"] == "kruskal"


def test_run_test_contingency_chi2():
    table = [[10, 20], [30, 40]]
    out = _arun(h.run_test_handler(test_name="chi2", data=table))
    assert out["success"] is True
    assert out["test_name"] == "chi2"


def test_run_test_contingency_fisher_exact():
    table = [[2, 5], [8, 3]]
    out = _arun(
        h.run_test_handler(
            test_name="fisher_exact", data=table, alternative="two-sided"
        )
    )
    assert out["success"] is True
    assert out["test_name"] == "fisher_exact"


def test_run_test_correlation_pearson():
    rng_c = np.random.default_rng(0)
    x = rng_c.normal(size=40)
    y = 0.5 * x + rng_c.normal(size=40, scale=0.3)
    out = _arun(
        h.run_test_handler(test_name="pearson", data=[x.tolist(), y.tolist()])
    )
    assert out["success"] is True
    assert out["test_name"] == "pearson"


def test_run_test_data_file_path(tmp_path):
    import pandas as _pd

    csv = tmp_path / "two_cols.csv"
    _pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 3.0, 5.0, 6.0, 7.0]}
    ).to_csv(csv, index=False)

    out = _arun(
        h.run_test_handler(
            test_name="pearson",
            data_file=str(csv),
            columns=["a", "b"],
        )
    )
    assert out["success"] is True
    assert out["test_name"] == "pearson"


def test_run_test_data_file_missing_column_errors(tmp_path):
    import pandas as _pd

    csv = tmp_path / "one_col.csv"
    _pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)

    out = _arun(
        h.run_test_handler(
            test_name="pearson",
            data_file=str(csv),
            columns=["a", "missing_col"],
        )
    )
    assert out["success"] is False
    assert "missing_col" in out["error"]


def test_run_test_dataframe_required_anova_rm():
    """Repeated-measures ANOVA: wide-format `data = [cond1, cond2, cond3]`,
    each cond a list of subject-aligned values."""
    rng_rm = np.random.default_rng(2)
    n_subj = 12
    # Same subjects across 3 conditions, with an increasing drift
    cond1 = rng_rm.normal(0, 1, n_subj).tolist()
    cond2 = (np.array(cond1) + rng_rm.normal(0.4, 0.3, n_subj)).tolist()
    cond3 = (np.array(cond1) + rng_rm.normal(0.8, 0.3, n_subj)).tolist()
    out = _arun(
        h.run_test_handler(test_name="anova_rm", data=[cond1, cond2, cond3])
    )
    assert out["success"] is True
    assert out["test_name"] == "anova_rm"


# ----- correct_pvalues_handler ----------------------------------------- #


def test_correct_pvalues_fdr_bh_returns_corrected():
    out = _arun(
        h.correct_pvalues_handler(
            pvalues=[0.001, 0.04, 0.03, 0.20, 0.005], method="fdr_bh"
        )
    )
    assert out["success"] is True
    assert out["method"] == "fdr_bh"
    assert out["n_tests"] == 5
    assert len(out["corrected_pvalues"]) == 5
    assert len(out["reject_null"]) == 5
    # corrected p's are monotonic-ish and never below original raw p
    raw_sorted = sorted(out["original_pvalues"])
    adj_sorted = sorted(out["corrected_pvalues"])
    assert all(a >= r - 1e-12 for r, a in zip(raw_sorted, adj_sorted))


def test_correct_pvalues_bonferroni():
    out = _arun(
        h.correct_pvalues_handler(
            pvalues=[0.01, 0.02, 0.05], method="bonferroni"
        )
    )
    assert out["success"] is True
    assert out["method"] == "bonferroni"
    # Bonferroni multiplies by n=3
    assert abs(out["corrected_pvalues"][0] - 0.03) < 1e-10


def test_correct_pvalues_holm():
    out = _arun(
        h.correct_pvalues_handler(
            pvalues=[0.001, 0.01, 0.05, 0.20], method="holm"
        )
    )
    assert out["success"] is True
    assert out["method"] == "holm"


def test_correct_pvalues_sidak():
    out = _arun(
        h.correct_pvalues_handler(
            pvalues=[0.001, 0.01, 0.05], method="sidak"
        )
    )
    assert out["success"] is True
    assert out["method"] == "sidak"


def test_correct_pvalues_fdr_by():
    out = _arun(
        h.correct_pvalues_handler(
            pvalues=[0.001, 0.01, 0.05, 0.20], method="fdr_by"
        )
    )
    assert out["success"] is True
    assert out["method"] == "fdr_by"


def test_correct_pvalues_unknown_method_falls_back_to_fdr_bh():
    """Handler's method_map.get(default='fdr_bh') maps unknown → fdr_bh."""
    out = _arun(
        h.correct_pvalues_handler(
            pvalues=[0.01, 0.02, 0.03], method="not_a_real_method"
        )
    )
    assert out["success"] is True
    # The handler echoes the requested method back even though
    # statsmodels was called with fdr_bh internally.
    assert out["method"] == "not_a_real_method"


def test_correct_pvalues_n_significant_counts_rejections_correctly():
    out = _arun(
        h.correct_pvalues_handler(
            pvalues=[0.001, 0.5, 0.6, 0.7], method="bonferroni", alpha=0.05
        )
    )
    assert out["success"] is True
    # Only 0.001 stays significant after Bonferroni (× 4)
    assert out["n_significant"] == 1
    assert sum(out["reject_null"]) == 1


# ----- correct_pvalues fallback path (statsmodels missing) ------------ #


def _arun_without_statsmodels(coro_factory):
    """Stub the statsmodels import inside `do_correct` to force the
    handler's no-statsmodels fallback path (lines 62-107)."""
    import sys as _sys

    saved = _sys.modules.get("statsmodels.stats.multitest")
    # Replace with a module-like object whose `multipletests` raises.
    bad = type(_sys)("statsmodels.stats.multitest")
    def _raise(*a, **k):
        raise ImportError("statsmodels stubbed out")
    bad.multipletests = _raise  # type: ignore[attr-defined]
    _sys.modules["statsmodels.stats.multitest"] = bad
    try:
        return _arun(coro_factory())
    finally:
        if saved is None:
            _sys.modules.pop("statsmodels.stats.multitest", None)
        else:
            _sys.modules["statsmodels.stats.multitest"] = saved


def test_correct_pvalues_fallback_bonferroni():
    out = _arun_without_statsmodels(
        lambda: h.correct_pvalues_handler(
            pvalues=[0.01, 0.02, 0.05], method="bonferroni"
        )
    )
    assert out["success"] is True
    assert out["method"] == "bonferroni"
    assert abs(out["corrected_pvalues"][0] - 0.03) < 1e-10


def test_correct_pvalues_fallback_holm():
    out = _arun_without_statsmodels(
        lambda: h.correct_pvalues_handler(
            pvalues=[0.001, 0.01, 0.05, 0.20], method="holm"
        )
    )
    assert out["success"] is True
    assert out["method"] == "holm"
    # Holm with n=4: smallest p * 4 = 0.004 → significant
    assert out["reject_null"][0] is True


def test_correct_pvalues_fallback_fdr_bh():
    out = _arun_without_statsmodels(
        lambda: h.correct_pvalues_handler(
            pvalues=[0.001, 0.01, 0.05, 0.20], method="fdr_bh"
        )
    )
    assert out["success"] is True
    assert out["method"] == "fdr_bh"


def test_correct_pvalues_fallback_sidak():
    out = _arun_without_statsmodels(
        lambda: h.correct_pvalues_handler(
            pvalues=[0.001, 0.01, 0.05], method="sidak"
        )
    )
    assert out["success"] is True
    assert out["method"] == "sidak"


def test_correct_pvalues_fallback_unknown_method_passes_through():
    """Fallback's `else` branch returns the raw p-values unchanged."""
    out = _arun_without_statsmodels(
        lambda: h.correct_pvalues_handler(
            pvalues=[0.01, 0.02, 0.05], method="not_a_real_method"
        )
    )
    assert out["success"] is True
    assert out["corrected_pvalues"] == [0.01, 0.02, 0.05]
