"""Tests for ``scitex_stats._mcp._handlers._corrections.correct_pvalues_handler``.

The handler is async; we wrap each call in ``asyncio.run`` so the suite
doesn't depend on pytest-asyncio (CI test envs without that plugin
otherwise see "async def functions are not natively supported").
"""

from __future__ import annotations

import asyncio

import pytest

from scitex_stats._mcp._handlers._corrections import correct_pvalues_handler


def _run(**kwargs):
    return asyncio.run(correct_pvalues_handler(**kwargs))


def test_returns_success_with_required_keys():
    out = _run(pvalues=[0.01, 0.02, 0.03, 0.04])
    assert out["success"] is True
    for k in (
        "original_pvalues",
        "corrected_pvalues",
        "reject_null",
        "n_significant",
        "n_tests",
    ):
        assert k in out, f"missing: {k}"


def test_n_tests_matches_input():
    pv = [0.01, 0.05, 0.1, 0.5, 0.9]
    out = _run(pvalues=pv)
    assert out["n_tests"] == 5


def test_corrected_array_same_length_as_input():
    pv = [0.01, 0.02, 0.03]
    out = _run(pvalues=pv)
    assert len(out["corrected_pvalues"]) == 3
    assert len(out["reject_null"]) == 3


def test_bonferroni_multiplies_by_n():
    """Bonferroni correction: corrected = min(1, p × n)."""
    pv = [0.01, 0.02, 0.03]
    out = _run(pvalues=pv, method="bonferroni")
    # 0.01 × 3 = 0.03
    assert abs(out["corrected_pvalues"][0] - 0.03) < 1e-9


def test_corrected_pvalues_at_least_originals():
    """Any correction must be ≥ the raw p-value (more conservative)."""
    pv = [0.001, 0.05, 0.5, 0.9]
    out = _run(pvalues=pv, method="bonferroni")
    for raw, corr in zip(pv, out["corrected_pvalues"]):
        assert corr >= raw - 1e-12


def test_n_significant_counts_rejections():
    pv = [0.001, 0.001, 0.5, 0.9]
    out = _run(pvalues=pv, method="bonferroni", alpha=0.05)
    # 0.001 × 4 = 0.004 < 0.05; 0.5 × 4 = 2.0 capped → not rejected.
    assert out["n_significant"] == 2
    assert out["reject_null"] == [True, True, False, False]


def test_fdr_bh_default():
    out = _run(pvalues=[0.01, 0.02, 0.03])
    # Default method should be FDR-BH; no exception, success path.
    assert out["success"] is True


@pytest.mark.parametrize(
    "method",
    ["bonferroni", "fdr_bh", "fdr_by", "holm", "sidak"],
)
def test_supported_methods(method):
    out = _run(pvalues=[0.001, 0.01, 0.05, 0.5], method=method)
    assert out["success"] is True


def test_alpha_changes_n_significant():
    pv = [0.01, 0.02, 0.04, 0.08]
    out_loose = _run(pvalues=pv, alpha=0.10)
    out_strict = _run(pvalues=pv, alpha=0.001)
    assert out_strict["n_significant"] <= out_loose["n_significant"]


def test_unknown_method_falls_back_to_fdr_bh():
    """Method map default routes anything unknown to fdr_bh."""
    pv = [0.01, 0.02, 0.03]
    out_explicit = _run(pvalues=pv, method="fdr_bh")
    out_unknown = _run(pvalues=pv, method="not-a-method")
    assert out_explicit["corrected_pvalues"] == out_unknown["corrected_pvalues"]


def test_original_pvalues_echoed():
    pv = [0.012, 0.034]
    out = _run(pvalues=pv)
    assert out["original_pvalues"] == pv
