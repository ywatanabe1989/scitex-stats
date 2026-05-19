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


def test_anova_dict_return_type_is_dict(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups, return_as="dict")

    # Assert
    assert isinstance(out, dict)


def test_anova_dataframe_return_type_is_dataframe(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups, return_as="dataframe")

    # Assert
    assert isinstance(out, pd.DataFrame)


def test_anova_dataframe_return_has_single_row(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups, return_as="dataframe")

    # Assert
    assert len(out) == 1


def test_anova_dict_has_required_fields(equal_means_groups):
    # Arrange
    required_keys = (
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
    )

    # Act
    out = _anova(groups=equal_means_groups)

    # Assert
    missing = [k for k in required_keys if k not in out]
    assert not missing, f"missing keys: {missing}"


def test_anova_test_method_label_mentions_anova(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups)

    # Assert
    assert "ANOVA" in out["test_method"].upper()


def test_anova_effect_size_metric_is_eta_squared(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups)

    # Assert
    assert "eta" in out["effect_size_metric"].lower()


def test_anova_h0_not_rejected_when_means_equal_significance_false(
    equal_means_groups,
):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups, alpha=0.05)

    # Assert
    assert out["significant"] is False


def test_anova_h0_not_rejected_when_means_equal_pvalue_above_alpha(
    equal_means_groups,
):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups, alpha=0.05)

    # Assert
    assert out["pvalue"] >= 0.05


def test_anova_h0_rejected_when_means_differ_significance_true(
    differing_means_groups,
):
    # Arrange
    groups = differing_means_groups

    # Act
    out = _anova(groups=groups, alpha=0.05)

    # Assert
    assert out["significant"] is True


def test_anova_h0_rejected_when_means_differ_pvalue_below_alpha(
    differing_means_groups,
):
    # Arrange
    groups = differing_means_groups

    # Act
    out = _anova(groups=groups, alpha=0.05)

    # Assert
    assert out["pvalue"] < 0.05


def test_anova_n_samples_matches_input_lengths(differing_means_groups):
    # Arrange
    expected = [len(g) for g in differing_means_groups]

    # Act
    out = _anova(groups=differing_means_groups)

    # Assert
    assert out["n_samples"] == expected


def test_anova_df_between_equals_k_minus_one(differing_means_groups):
    # Arrange
    expected = len(differing_means_groups) - 1

    # Act
    out = _anova(groups=differing_means_groups)

    # Assert
    assert out["df_between"] == expected


def test_anova_df_within_equals_n_minus_k(differing_means_groups):
    # Arrange
    total = sum(len(g) for g in differing_means_groups)
    expected = total - len(differing_means_groups)

    # Act
    out = _anova(groups=differing_means_groups)

    # Assert
    assert out["df_within"] == expected


def test_anova_effect_size_lies_in_unit_interval(differing_means_groups):
    # Arrange
    groups = differing_means_groups

    # Act
    out = _anova(groups=groups)

    # Assert
    assert 0.0 <= out["effect_size"] <= 1.0


def test_anova_pvalue_lies_in_unit_interval(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups)

    # Assert
    assert 0.0 <= out["pvalue"] <= 1.0


def test_anova_default_var_names_match_group_count(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups)

    # Assert
    assert len(out["var_names"]) == len(groups)


def test_anova_default_var_names_use_group_prefix(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups)

    # Assert
    assert all(n.startswith("Group") for n in out["var_names"])


def test_anova_custom_var_names_preserved_in_output(equal_means_groups):
    # Arrange
    custom = ["alpha", "beta", "gamma"]

    # Act
    out = _anova(groups=equal_means_groups, var_names=custom)

    # Assert
    assert out["var_names"] == custom


def test_anova_dataframe_input_via_value_and_group_cols_group_count():
    # Arrange
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

    # Act
    out = _anova(data=df, value_col="score", group_col="arm")

    # Assert
    assert out["n_groups"] == 3


def test_anova_dataframe_input_via_value_and_group_cols_detects_difference():
    # Arrange
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

    # Act
    out = _anova(data=df, value_col="score", group_col="arm")

    # Assert
    assert out["significant"] is True


def test_anova_assumptions_check_returns_bool_or_dict(equal_means_groups):
    # Arrange
    groups = equal_means_groups

    # Act
    out = _anova(groups=groups, check_assumptions=True)

    # Assert
    assert isinstance(out["assumptions_met"], (bool, dict))


def test_anova_raises_when_only_one_group_supplied():
    # Arrange
    rng = np.random.default_rng(seed=0)
    one_group = [rng.normal(size=20)]
    expected_exc = (ValueError, AssertionError, IndexError)

    # Act
    raises_ctx = pytest.raises(expected_exc)

    # Assert
    with raises_ctx:
        _anova(groups=one_group)
