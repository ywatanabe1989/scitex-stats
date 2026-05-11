"""Tests for ``scitex_stats.tests.normality._test_shapiro.test_shapiro``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scitex_stats.tests.normality._test_shapiro import test_shapiro as _sw


def test_returns_dict():
    rng = np.random.default_rng(0)
    out = _sw(x=rng.normal(size=100), return_as="dict")
    assert isinstance(out, dict)


def test_returns_dataframe():
    rng = np.random.default_rng(0)
    out = _sw(x=rng.normal(size=100), return_as="dataframe")
    assert isinstance(out, pd.DataFrame) and len(out) == 1


def test_required_keys():
    rng = np.random.default_rng(0)
    out = _sw(x=rng.normal(size=100))
    for k in (
        "test_method",
        "statistic",
        "pvalue",
        "stars",
        "significant",
        "normal",
        "n",
    ):
        assert k in out, f"missing: {k}"


def test_method_label():
    rng = np.random.default_rng(0)
    out = _sw(x=rng.normal(size=50))
    assert "shapiro" in out["test_method"].lower()


def test_normal_true_for_gaussian_sample():
    rng = np.random.default_rng(seed=42)
    out = _sw(x=rng.normal(loc=0, scale=1, size=200), alpha=0.05)
    assert out["normal"] is True
    assert out["significant"] is False


def test_normal_false_for_uniform_sample():
    rng = np.random.default_rng(seed=42)
    out = _sw(x=rng.uniform(low=-1, high=1, size=200), alpha=0.05)
    assert out["normal"] is False
    assert out["significant"] is True


def test_normal_false_for_exponential_sample():
    rng = np.random.default_rng(seed=42)
    out = _sw(x=rng.exponential(scale=1.0, size=200), alpha=0.05)
    assert out["normal"] is False


def test_w_statistic_in_unit_interval():
    rng = np.random.default_rng(0)
    out = _sw(x=rng.normal(size=100))
    assert 0.0 <= out["statistic"] <= 1.0


def test_pvalue_in_unit_interval():
    rng = np.random.default_rng(0)
    out = _sw(x=rng.normal(size=100))
    assert 0.0 <= out["pvalue"] <= 1.0


def test_n_matches_sample_size():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=137)
    out = _sw(x=sample)
    assert out["n"] == 137


def test_dataframe_input_via_column_name():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"weight": rng.normal(size=100)})
    out = _sw(x="weight", data=df)
    assert out["n"] == 100


def test_pandas_series_input():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=100))
    out = _sw(x=s)
    assert out["n"] == 100


def test_alpha_threshold_changes_significance():
    """Borderline normal data: tighter alpha → less likely to reject."""
    rng = np.random.default_rng(seed=1)
    # Mildly skewed mixture; result is borderline.
    sample = np.concatenate([rng.normal(0, 1, 50), rng.normal(0.5, 1.2, 50)])
    out_strict = _sw(x=sample, alpha=0.001)
    out_loose = _sw(x=sample, alpha=0.5)
    # Strict is at most as likely to flag as loose (monotonic in alpha).
    if out_strict["significant"]:
        assert out_loose["significant"]
