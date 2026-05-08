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


def test_returns_dict(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y, return_as="dict")
    assert isinstance(out, dict)


def test_returns_dataframe(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y, return_as="dataframe")
    assert isinstance(out, pd.DataFrame) and len(out) == 1


def test_required_keys(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y)
    for k in ("test_method", "statistic", "pvalue", "stars", "significant"):
        assert k in out, f"missing: {k}"


def test_method_label(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y)
    assert "pearson" in out["test_method"].lower()


def test_r_close_to_one_for_positive_linear(perfectly_correlated):
    x, y = perfectly_correlated
    out = _pearson(x=x, y=y)
    assert out["statistic"] > 0.99


def test_r_close_to_minus_one_for_negative_linear(perfectly_anticorrelated):
    x, y = perfectly_anticorrelated
    out = _pearson(x=x, y=y)
    assert out["statistic"] < -0.99


def test_r_in_unit_interval(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y)
    assert -1.0 <= out["statistic"] <= 1.0


def test_pvalue_in_unit_interval(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y)
    assert 0.0 <= out["pvalue"] <= 1.0


def test_significant_for_strong_correlation(perfectly_correlated):
    x, y = perfectly_correlated
    out = _pearson(x=x, y=y, alpha=0.05)
    assert out["significant"] is True


def test_not_significant_for_independent_samples(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y, alpha=0.05)
    # Independent N(0,1) vs N(0,1) at n=500 → p typically > 0.05.
    assert out["significant"] is False


def test_dataframe_input_via_columns():
    rng = np.random.default_rng(seed=2)
    x = rng.uniform(0, 10, 200)
    df = pd.DataFrame({"x": x, "y": x * 0.5 + rng.normal(0, 0.1, 200)})
    out = _pearson(x="x", y="y", data=df)
    assert out["statistic"] > 0.95


def test_pandas_series_input(perfectly_correlated):
    x, y = perfectly_correlated
    out = _pearson(x=pd.Series(x), y=pd.Series(y))
    assert out["statistic"] > 0.99


def test_alternative_greater(perfectly_correlated):
    x, y = perfectly_correlated
    out_two = _pearson(x=x, y=y, alternative="two-sided")
    out_greater = _pearson(x=x, y=y, alternative="greater")
    # One-sided "greater" against r > 0 should be (at most) the two-sided p.
    assert out_greater["pvalue"] <= out_two["pvalue"] + 1e-12


def test_alternative_less_for_anticorrelated(perfectly_anticorrelated):
    x, y = perfectly_anticorrelated
    out = _pearson(x=x, y=y, alternative="less", alpha=0.05)
    assert out["significant"] is True


def test_var_labels_passthrough(uncorrelated):
    x, y = uncorrelated
    out = _pearson(x=x, y=y, var_x="weight_kg", var_y="height_cm")
    assert out["var_x"] == "weight_kg"
    assert out["var_y"] == "height_cm"


def test_mismatched_lengths_raises():
    rng = np.random.default_rng(0)
    with pytest.raises((ValueError, AssertionError)):
        _pearson(x=rng.normal(size=10), y=rng.normal(size=11))
