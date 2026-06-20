"""Tests for ``scitex_stats.tests.correlation._test_pearson.test_pearson``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scitex_stats.tests.correlation._test_pearson import test_pearson as _pearson


@pytest.fixture
def perfectly_correlated():
    """y = 2x + small noise → Pearson r ≈ 1."""
    rng = np.random.default_rng(seed=0)
    x = rng.uniform(-5, 5, 200)
    y = 2 * x + rng.normal(0, 1e-6, 200)
    return x, y


@pytest.fixture
def perfectly_anticorrelated():
    """y = -2x → Pearson r ≈ -1."""
    rng = np.random.default_rng(seed=0)
    x = rng.uniform(-5, 5, 200)
    y = -2 * x + rng.normal(0, 1e-6, 200)
    return x, y


@pytest.fixture
def uncorrelated():
    """Two independent Gaussians → r ≈ 0."""
    rng = np.random.default_rng(seed=0)
    return rng.normal(size=500), rng.normal(size=500)


def test_returns_dict_case(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y, return_as="dict")
    # Assert
    assert isinstance(out, dict)


def test_returns_dataframe_case(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y, return_as="dataframe")
    # Assert
    assert isinstance(out, pd.DataFrame) and len(out) == 1


def test_required_keys_test_method_statistic_pvalue(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y)
    # Assert
    for k in ("test_method", "statistic", "pvalue", "stars", "significant"):
        assert k in out, f"missing: {k}"


def test_method_label_pearson_lower_test(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y)
    # Assert
    assert "pearson" in out["test_method"].lower()


def test_r_close_to_one_for_positive_linear(perfectly_correlated):
    # Arrange
    x, y = perfectly_correlated
    # Act
    out = _pearson(x=x, y=y)
    # Assert
    assert out["statistic"] > 0.99


def test_r_close_to_minus_one_for_negative_linear(perfectly_anticorrelated):
    # Arrange
    x, y = perfectly_anticorrelated
    # Act
    out = _pearson(x=x, y=y)
    # Assert
    assert out["statistic"] < -0.99


def test_r_in_unit_interval(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y)
    # Assert
    assert -1.0 <= out["statistic"] <= 1.0


def test_pvalue_in_unit_interval(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y)
    # Assert
    assert 0.0 <= out["pvalue"] <= 1.0


def test_significant_for_strong_correlation(perfectly_correlated):
    # Arrange
    x, y = perfectly_correlated
    # Act
    out = _pearson(x=x, y=y, alpha=0.05)
    # Assert
    assert out["significant"] is True


def test_not_significant_for_independent_samples(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y, alpha=0.05)
    # Assert
    assert out["significant"] is False


def test_dataframe_input_via_columns():
    # Arrange
    rng = np.random.default_rng(seed=2)
    x = rng.uniform(0, 10, 200)
    df = pd.DataFrame({"x": x, "y": x * 0.5 + rng.normal(0, 0.1, 200)})
    # Act
    out = _pearson(x="x", y="y", data=df)
    # Assert
    assert out["statistic"] > 0.95


def test_pandas_series_input(perfectly_correlated):
    # Arrange
    x, y = perfectly_correlated
    # Act
    out = _pearson(x=pd.Series(x), y=pd.Series(y))
    # Assert
    assert out["statistic"] > 0.99


def test_alternative_greater_out_greater_pvalue_out_two(perfectly_correlated):
    # Arrange
    x, y = perfectly_correlated
    out_two = _pearson(x=x, y=y, alternative="two-sided")
    # Act
    out_greater = _pearson(x=x, y=y, alternative="greater")
    # Assert
    assert out_greater["pvalue"] <= out_two["pvalue"] + 1e-12


def test_alternative_less_for_anticorrelated(perfectly_anticorrelated):
    # Arrange
    x, y = perfectly_anticorrelated
    # Act
    out = _pearson(x=x, y=y, alternative="less", alpha=0.05)
    # Assert
    assert out["significant"] is True


def test_var_labels_passthrough_weight_kg(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y, var_x="weight_kg", var_y="height_cm")
    # Assert
    assert out["var_x"] == "weight_kg"

def test_var_labels_passthrough_height_cm(uncorrelated):
    # Arrange
    x, y = uncorrelated
    # Act
    out = _pearson(x=x, y=y, var_x="weight_kg", var_y="height_cm")
    # Assert
    assert out["var_y"] == "height_cm"


def test_mismatched_lengths_raises():
    # Arrange
    # Act
    rng = np.random.default_rng(0)
    # Assert
    with pytest.raises((ValueError, AssertionError)):
        _pearson(x=rng.normal(size=10), y=rng.normal(size=11))
