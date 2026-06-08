"""Tests for ``scitex_stats.tests.agreement._test_icc.test_icc``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scitex_stats.tests.agreement._test_icc import (
    interpret_icc,
    test_icc as _icc,
)


@pytest.fixture
def perfect_agreement():
    """Identical raters → ICC ≈ 1."""
    rng = np.random.default_rng(0)
    subj = rng.normal(size=20)
    return np.tile(subj.reshape(-1, 1), (1, 5))


@pytest.fixture
def random_no_agreement():
    """Independent columns → ICC ≈ 0 (often slightly negative)."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(20, 5))


def test_perfect_agreement_returns_icc3k_near_one(perfect_agreement):
    # Arrange
    mat = perfect_agreement
    # Act
    out = _icc(mat, form="3,k")
    # Assert
    assert out["statistic"] > 0.99


def test_random_matrix_returns_low_icc3k(random_no_agreement):
    # Arrange
    mat = random_no_agreement
    # Act
    out = _icc(mat, form="3,k")
    # Assert
    assert out["statistic"] < 0.5


def test_icc_returns_required_keys(random_no_agreement):
    # Arrange
    mat = random_no_agreement
    # Act
    out = _icc(mat, form="3,k")
    # Assert
    for k in ("statistic", "pvalue", "df1", "df2", "F",
              "n", "k", "formatted"):
        assert k in out


def test_icc_returns_all_six_forms(random_no_agreement):
    # Arrange
    mat = random_no_agreement
    # Act
    out = _icc(mat, form="3,k")
    # Assert
    for form_key in ("ICC(1,1)", "ICC(2,1)", "ICC(3,1)",
                     "ICC(1,k)", "ICC(2,k)", "ICC(3,k)"):
        assert form_key in out


def test_form_selection_changes_top_level_statistic(random_no_agreement):
    # Arrange — non-degenerate matrix so the single / avg forms differ
    mat = random_no_agreement
    # Act
    out_single = _icc(mat, form="3,1")
    out_avg = _icc(mat, form="3,k")
    # Assert
    assert out_single["statistic"] != pytest.approx(out_avg["statistic"])


def test_unknown_form_raises_value_error(random_no_agreement):
    # Arrange
    mat = random_no_agreement
    # Act
    # (call performed inside the pytest.raises context below)
    # Assert
    with pytest.raises(ValueError):
        _icc(mat, form="bogus")


def test_one_dim_input_raises_value_error():
    # Arrange
    bad = np.arange(10.0)
    # Act
    # (call performed inside the pytest.raises context below)
    # Assert
    with pytest.raises(ValueError):
        _icc(bad)


def test_returns_dataframe_when_requested(random_no_agreement):
    # Arrange
    mat = random_no_agreement
    # Act
    out = _icc(mat, form="3,k", return_as="dataframe")
    # Assert
    assert isinstance(out, pd.DataFrame)


def test_long_format_dataframe_pivots_correctly():
    # Arrange
    long_df = pd.DataFrame({
        "subj": ["a", "a", "b", "b", "c", "c"],
        "rater": ["r1", "r2", "r1", "r2", "r1", "r2"],
        "score": [1.0, 1.2, 2.0, 1.9, 3.0, 3.1],
    })
    # Act
    out = _icc(long_df, form="3,k", subj_col="subj",
               rater_col="rater", score_col="score")
    # Assert
    assert out["n"] == 3 and out["k"] == 2


def test_interpret_icc_below_point_five_is_poor():
    # Arrange
    icc_value = 0.4
    # Act
    label = interpret_icc(icc_value)
    # Assert
    assert label == "poor"


def test_interpret_icc_below_point_seventy_five_is_moderate():
    # Arrange
    icc_value = 0.6
    # Act
    label = interpret_icc(icc_value)
    # Assert
    assert label == "moderate"


def test_interpret_icc_below_point_ninety_is_good():
    # Arrange
    icc_value = 0.8
    # Act
    label = interpret_icc(icc_value)
    # Assert
    assert label == "good"


def test_interpret_icc_at_or_above_point_ninety_is_excellent():
    # Arrange
    icc_value = 0.95
    # Act
    label = interpret_icc(icc_value)
    # Assert
    assert label == "excellent"
