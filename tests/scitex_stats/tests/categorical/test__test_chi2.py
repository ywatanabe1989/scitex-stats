"""Tests for ``scitex_stats.tests.categorical._test_chi2.test_chi2``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scitex_stats.tests.categorical._test_chi2 import test_chi2 as _chi2


@pytest.fixture
def independent_table():
    """2×2 table where variables are independent (proportions equal)."""
    return np.array([[50, 50], [50, 50]])


@pytest.fixture
def dependent_table():
    """3×2 table with strong association."""
    return np.array([[80, 20], [50, 50], [20, 80]])


def test_returns_dict_case(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table, return_as="dict")
    # Assert
    assert isinstance(out, dict)


def test_returns_dataframe_case(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table, return_as="dataframe")
    # Assert
    assert isinstance(out, pd.DataFrame) and len(out) == 1


def test_required_keys_test_method_statistic_pvalue(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table)
    # Assert
    for k in (
        "test_method",
        "statistic",
        "pvalue",
        "stars",
        "significant",
        "df",
        "effect_size",
        "effect_size_metric",
        "n",
    ):
        assert k in out, f"missing: {k}"


def test_method_label_chi_lower_test(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table)
    # Assert
    assert "chi" in out["test_method"].lower()


def test_effect_size_is_cramers_v(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table)
    # Assert
    assert (
        "cramér" in out["effect_size_metric"].lower()
        or "cramer" in out["effect_size_metric"].lower()
    )


def test_h0_holds_for_independent_table_significant(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table, alpha=0.05)
    # Assert
    assert out["significant"] is False

def test_h0_holds_for_independent_table_pvalue(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table, alpha=0.05)
    # Assert
    assert out["pvalue"] >= 0.05


def test_h0_rejected_for_associated_table_significant(dependent_table):
    # Arrange
    # Act
    out = _chi2(observed=dependent_table, alpha=0.05)
    # Assert
    assert out["significant"] is True

def test_h0_rejected_for_associated_table_pvalue(dependent_table):
    # Arrange
    # Act
    out = _chi2(observed=dependent_table, alpha=0.05)
    # Assert
    assert out["pvalue"] < 0.05


def test_n_equals_sum_of_table():
    # Arrange
    table = np.array([[10, 20], [30, 40]])
    # Act
    out = _chi2(observed=table)
    # Assert
    assert out["n"] == 100


def test_df_for_2x2_is_1():
    # Arrange
    # Act
    out = _chi2(observed=np.array([[10, 20], [30, 40]]))
    # Assert
    assert out["df"] == 1


def test_df_for_3x2_is_2(dependent_table):
    # Arrange
    # Act
    out = _chi2(observed=dependent_table)
    # Assert
    assert out["df"] == 2


def test_cramers_v_in_unit_interval(dependent_table):
    # Arrange
    # Act
    out = _chi2(observed=dependent_table)
    # Assert
    assert 0.0 <= out["effect_size"] <= 1.0


def test_cramers_v_smaller_for_independent_than_dependent(
    independent_table, dependent_table
):
    # Arrange
    out_ind = _chi2(observed=independent_table)
    # Act
    out_dep = _chi2(observed=dependent_table)
    # Assert
    assert out_ind["effect_size"] < out_dep["effect_size"]


def test_dataframe_input_passthrough_var_names():
    # Arrange
    df = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=["young", "old"],
        columns=["yes", "no"],
    )
    # Act
    out = _chi2(observed=df)
    # Assert
    assert out["var_row"] in (df.index.name, "row_variable", None) or isinstance(
        out["var_row"], str
    )


def test_var_row_and_col_overrides_age():
    # Arrange
    # Act
    out = _chi2(
        observed=np.array([[10, 20], [30, 40]]),
        var_row="age",
        var_col="response",
    )
    # Assert
    assert out["var_row"] == "age"

def test_var_row_and_col_overrides_response():
    # Arrange
    # Act
    out = _chi2(
        observed=np.array([[10, 20], [30, 40]]),
        var_row="age",
        var_col="response",
    )
    # Assert
    assert out["var_col"] == "response"


def test_pvalue_in_unit_interval(independent_table):
    # Arrange
    # Act
    out = _chi2(observed=independent_table)
    # Assert
    assert 0.0 <= out["pvalue"] <= 1.0


def test_yates_correction_changes_2x2_statistic():
    """For 2×2 tables, correction=True applies Yates' adjustment."""
    # Arrange
    table = np.array([[12, 18], [25, 5]])
    out_corr = _chi2(observed=table, correction=True)
    # Act
    out_raw = _chi2(observed=table, correction=False)
    # Assert
    assert out_corr["statistic"] <= out_raw["statistic"]


def test_zero_cells_handled_statistic():
    """Tables with zero cells should not crash; degenerate but not an error."""
    # Arrange
    table = np.array([[10, 0, 5], [3, 7, 0]])
    # Act
    out = _chi2(observed=table)
    # Assert
    assert "statistic" in out

def test_zero_cells_handled_isfinite_statistic():
    """Tables with zero cells should not crash; degenerate but not an error."""
    # Arrange
    table = np.array([[10, 0, 5], [3, 7, 0]])
    # Act
    out = _chi2(observed=table)
    # Assert
    assert np.isfinite(out["statistic"])
