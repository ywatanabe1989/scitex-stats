"""Tests for ``scitex_stats.tests.parametric._test_anova.test_anova``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Aliased to avoid pytest auto-collecting the imported function name
# (`test_anova` would otherwise be picked up as a test itself).
from scitex_stats.tests.parametric._test_anova import test_anova as _anova


@pytest.fixture
def equal_means_groups():
    """Three groups drawn from the same distribution; H0 should hold."""
    rng = np.random.default_rng(seed=42)
    return [rng.normal(loc=0.0, scale=1.0, size=30) for _ in range(3)]


@pytest.fixture
def differing_means_groups():
    """Three groups with shifted means; H0 should be rejected."""
    rng = np.random.default_rng(seed=42)
    return [
        rng.normal(loc=0.0, scale=1.0, size=30),
        rng.normal(loc=1.5, scale=1.0, size=30),
        rng.normal(loc=3.0, scale=1.0, size=30),
    ]


def test_returns_dict_when_return_as_dict(equal_means_groups):
    out = _anova(groups=equal_means_groups, return_as="dict")
    assert isinstance(out, dict)


def test_returns_dataframe_when_return_as_dataframe(equal_means_groups):
    out = _anova(groups=equal_means_groups, return_as="dataframe")
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1


def test_dict_has_required_fields(equal_means_groups):
    out = _anova(groups=equal_means_groups)
    for key in (
        "test_method",
        "statistic",
        "pvalue",
        "stars",
        "significant",
        "effect_size",
        "effect_size_metric",
        "n_groups",
        "n_samples",
        "df_between",
        "df_within",
    ):
        assert key in out, f"missing key: {key}"


def test_test_method_label(equal_means_groups):
    out = _anova(groups=equal_means_groups)
    assert "ANOVA" in out["test_method"].upper()


def test_effect_size_metric_is_eta_squared(equal_means_groups):
    out = _anova(groups=equal_means_groups)
    assert "eta" in out["effect_size_metric"].lower()


def test_h0_not_rejected_for_equal_means(equal_means_groups):
    out = _anova(groups=equal_means_groups, alpha=0.05)
    assert out["significant"] is False
    assert out["pvalue"] >= 0.05


def test_h0_rejected_for_clearly_differing_means(differing_means_groups):
    out = _anova(groups=differing_means_groups, alpha=0.05)
    assert out["significant"] is True
    assert out["pvalue"] < 0.05


def test_n_samples_match_input_lengths(differing_means_groups):
    out = _anova(groups=differing_means_groups)
    assert out["n_samples"] == [len(g) for g in differing_means_groups]


def test_df_between_equals_k_minus_1(differing_means_groups):
    out = _anova(groups=differing_means_groups)
    assert out["df_between"] == len(differing_means_groups) - 1


def test_df_within_equals_N_minus_k(differing_means_groups):
    out = _anova(groups=differing_means_groups)
    total = sum(len(g) for g in differing_means_groups)
    assert out["df_within"] == total - len(differing_means_groups)


def test_effect_size_in_unit_interval(differing_means_groups):
    out = _anova(groups=differing_means_groups)
    assert 0.0 <= out["effect_size"] <= 1.0


def test_pvalue_in_unit_interval(equal_means_groups):
    out = _anova(groups=equal_means_groups)
    assert 0.0 <= out["pvalue"] <= 1.0


def test_var_names_default_when_omitted(equal_means_groups):
    out = _anova(groups=equal_means_groups)
    names = out["var_names"]
    assert len(names) == len(equal_means_groups)
    assert all(n.startswith("Group") for n in names)


def test_var_names_used_when_given(equal_means_groups):
    out = _anova(
        groups=equal_means_groups,
        var_names=["alpha", "beta", "gamma"],
    )
    assert out["var_names"] == ["alpha", "beta", "gamma"]


def test_dataframe_input_via_value_and_group_cols():
    """Seaborn-style: pass DataFrame + value_col + group_col."""
    rng = np.random.default_rng(seed=7)
    df = pd.DataFrame(
        {
            "score": np.concatenate(
                [
                    rng.normal(0, 1, 20),
                    rng.normal(2, 1, 20),
                    rng.normal(4, 1, 20),
                ]
            ),
            "arm": ["control"] * 20 + ["low"] * 20 + ["high"] * 20,
        }
    )
    out = _anova(data=df, value_col="score", group_col="arm")
    assert out["n_groups"] == 3
    assert out["significant"] is True


def test_assumptions_check_returns_bool_or_dict(equal_means_groups):
    out = _anova(groups=equal_means_groups, check_assumptions=True)
    # `assumptions_met` should be a boolean or a dict with details.
    val = out["assumptions_met"]
    assert isinstance(val, (bool, dict))


def test_minimum_two_groups_required():
    """One-group input is invalid for one-way ANOVA."""
    rng = np.random.default_rng(seed=0)
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _anova(groups=[rng.normal(size=20)])
