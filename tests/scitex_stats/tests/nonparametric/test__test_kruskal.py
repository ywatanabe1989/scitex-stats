"""Tests for ``scitex_stats.tests.nonparametric._test_kruskal.test_kruskal``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scitex_stats.tests.nonparametric._test_kruskal import test_kruskal as _kw


@pytest.fixture
def equal_dist_groups():
    rng = np.random.default_rng(seed=11)
    return [rng.normal(0.0, 1.0, size=30) for _ in range(3)]


@pytest.fixture
def shifted_groups():
    rng = np.random.default_rng(seed=11)
    return [
        rng.normal(0.0, 1.0, size=30),
        rng.normal(2.5, 1.0, size=30),
        rng.normal(5.0, 1.0, size=30),
    ]


def test_returns_dict_case(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups, return_as="dict")
    # Assert
    assert isinstance(out, dict)


def test_returns_dataframe_case(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups, return_as="dataframe")
    # Assert
    assert isinstance(out, pd.DataFrame) and len(out) == 1


def test_required_keys_test_method_statistic_pvalue(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups)
    # Assert
    for k in (
        "test_method",
        "statistic",
        "pvalue",
        "stars",
        "significant",
        "effect_size",
        "effect_size_metric",
        "n_groups",
        "n_samples",
    ):
        assert k in out, f"missing: {k}"


def test_method_label_kruskal_lower_test(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups)
    # Assert
    assert "kruskal" in out["test_method"].lower()


def test_effect_size_is_epsilon_squared(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups)
    # Assert
    assert "epsilon" in out["effect_size_metric"].lower()


def test_h0_holds_for_equal_distributions_significant(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups, alpha=0.05)
    # Assert
    assert out["significant"] is False

def test_h0_holds_for_equal_distributions_pvalue(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups, alpha=0.05)
    # Assert
    assert out["pvalue"] >= 0.05


def test_h0_rejected_for_shifted_distributions_significant(shifted_groups):
    # Arrange
    # Act
    out = _kw(groups=shifted_groups, alpha=0.05)
    # Assert
    assert out["significant"] is True

def test_h0_rejected_for_shifted_distributions_pvalue(shifted_groups):
    # Arrange
    # Act
    out = _kw(groups=shifted_groups, alpha=0.05)
    # Assert
    assert out["pvalue"] < 0.05


def test_pvalue_in_unit_interval(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups)
    # Assert
    assert 0.0 <= out["pvalue"] <= 1.0


def test_n_groups_reflects_input(shifted_groups):
    # Arrange
    # Act
    out = _kw(groups=shifted_groups)
    # Assert
    assert out["n_groups"] == len(shifted_groups)


def test_n_samples_reflects_input(shifted_groups):
    # Arrange
    # Act
    out = _kw(groups=shifted_groups)
    # Assert
    assert out["n_samples"] == [len(g) for g in shifted_groups]


def test_var_names_default(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups)
    # Assert
    assert len(out["var_names"]) == len(equal_dist_groups)


def test_var_names_passthrough(equal_dist_groups):
    # Arrange
    # Act
    out = _kw(groups=equal_dist_groups, var_names=["a", "b", "c"])
    # Assert
    assert out["var_names"] == ["a", "b", "c"]


def test_dataframe_input_groups():
    # Arrange
    rng = np.random.default_rng(seed=3)
    df = pd.DataFrame(
        {
            "y": np.concatenate(
                [rng.normal(0, 1, 25), rng.normal(3, 1, 25), rng.normal(6, 1, 25)]
            ),
            "g": ["A"] * 25 + ["B"] * 25 + ["C"] * 25,
        }
    )
    # Act
    out = _kw(data=df, value_col="y", group_col="g")
    # Assert
    assert out["n_groups"] == 3

def test_dataframe_input_significant():
    # Arrange
    rng = np.random.default_rng(seed=3)
    df = pd.DataFrame(
        {
            "y": np.concatenate(
                [rng.normal(0, 1, 25), rng.normal(3, 1, 25), rng.normal(6, 1, 25)]
            ),
            "g": ["A"] * 25 + ["B"] * 25 + ["C"] * 25,
        }
    )
    # Act
    out = _kw(data=df, value_col="y", group_col="g")
    # Assert
    assert out["significant"] is True


def test_robust_to_outliers():
    """KW should still flag a real shift even with one extreme outlier."""
    # Arrange
    rng = np.random.default_rng(seed=5)
    g1 = rng.normal(0.0, 1.0, size=30)
    g2 = rng.normal(2.0, 1.0, size=30)
    g2[0] = 100.0  # huge outlier
    # Act
    out = _kw(groups=[g1, g2], alpha=0.05)
    # Assert
    assert out["significant"] is True
