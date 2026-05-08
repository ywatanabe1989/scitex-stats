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
