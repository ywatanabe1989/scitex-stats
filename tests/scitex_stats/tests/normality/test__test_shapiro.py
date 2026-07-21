"""Tests for ``scitex_stats.tests.normality._test_shapiro.test_shapiro``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scitex_stats.tests.normality._test_shapiro import test_shapiro as _sw


def test_returns_dict_case():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    out = _sw(x=rng.normal(size=100), return_as="dict")
    # Assert
    assert isinstance(out, dict)


def test_returns_dataframe_case():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    out = _sw(x=rng.normal(size=100), return_as="dataframe")
    # Assert
    assert isinstance(out, pd.DataFrame) and len(out) == 1


def test_required_keys_test_method_statistic_pvalue():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    out = _sw(x=rng.normal(size=100))
    # Assert
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


def test_method_label_shapiro_lower_test():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    out = _sw(x=rng.normal(size=50))
    # Assert
    assert "shapiro" in out["test_method"].lower()


def test_normal_true_for_gaussian_sample_case_1():
    # Arrange
    rng = np.random.default_rng(seed=42)
    # Act
    out = _sw(x=rng.normal(loc=0, scale=1, size=200), alpha=0.05)
    # Assert
    assert out["normal"] is True

def test_normal_true_for_gaussian_sample_significant():
    # Arrange
    rng = np.random.default_rng(seed=42)
    # Act
    out = _sw(x=rng.normal(loc=0, scale=1, size=200), alpha=0.05)
    # Assert
    assert out["significant"] is False


def test_normal_false_for_uniform_sample_case_1():
    # Arrange
    rng = np.random.default_rng(seed=42)
    # Act
    out = _sw(x=rng.uniform(low=-1, high=1, size=200), alpha=0.05)
    # Assert
    assert out["normal"] is False

def test_normal_false_for_uniform_sample_significant():
    # Arrange
    rng = np.random.default_rng(seed=42)
    # Act
    out = _sw(x=rng.uniform(low=-1, high=1, size=200), alpha=0.05)
    # Assert
    assert out["significant"] is True


def test_normal_false_for_exponential_sample():
    # Arrange
    rng = np.random.default_rng(seed=42)
    # Act
    out = _sw(x=rng.exponential(scale=1.0, size=200), alpha=0.05)
    # Assert
    assert out["normal"] is False


def test_w_statistic_in_unit_interval():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    out = _sw(x=rng.normal(size=100))
    # Assert
    assert 0.0 <= out["statistic"] <= 1.0


def test_pvalue_in_unit_interval():
    # Arrange
    rng = np.random.default_rng(0)
    # Act
    out = _sw(x=rng.normal(size=100))
    # Assert
    assert 0.0 <= out["pvalue"] <= 1.0


def test_n_matches_sample_size():
    # Arrange
    rng = np.random.default_rng(0)
    sample = rng.normal(size=137)
    # Act
    out = _sw(x=sample)
    # Assert
    assert out["n"] == 137


def test_dataframe_input_via_column_name():
    # Arrange
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"weight": rng.normal(size=100)})
    # Act
    out = _sw(x="weight", data=df)
    # Assert
    assert out["n"] == 100


def test_pandas_series_input():
    # Arrange
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=100))
    # Act
    out = _sw(x=s)
    # Assert
    assert out["n"] == 100


def test_alpha_threshold_changes_significance():
    """Borderline normal data: tighter alpha → less likely to reject."""
    # Arrange
    rng = np.random.default_rng(seed=1)
    sample = np.concatenate([rng.normal(0, 1, 50), rng.normal(0.5, 1.2, 50)])
    out_strict = _sw(x=sample, alpha=0.001)
    # Act
    out_loose = _sw(x=sample, alpha=0.5)
    # Assert
    if out_strict["significant"]:
        assert out_loose["significant"]
