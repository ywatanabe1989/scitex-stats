"""Tests for
``scitex_stats.tests.agreement._test_kendalls_w.test_kendalls_w``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scitex_stats.tests.agreement._test_kendalls_w import (
    interpret_kendalls_w,
    test_kendalls_w as _w,
)


@pytest.fixture
def perfect_agreement():
    """k raters all rank the n subjects identically → W = 1."""
    # Arrange — same order across columns
    ranks = np.tile(np.arange(10).reshape(-1, 1), (1, 5)).astype(float)
    return ranks


@pytest.fixture
def random_no_agreement():
    """k random columns → W near 0 in expectation."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 6))


def test_perfect_agreement_returns_W_equal_one(perfect_agreement):
    # Arrange
    mat = perfect_agreement
    # Act
    out = _w(mat)
    # Assert
    assert pytest.approx(out["W"], abs=1e-9) == 1.0


def test_random_columns_return_low_W(random_no_agreement):
    # Arrange
    mat = random_no_agreement
    # Act
    out = _w(mat)
    # Assert
    assert 0.0 <= out["W"] <= 0.4


def test_W_in_unit_interval_on_random_matrix():
    # Arrange
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(15, 4))
    # Act
    out = _w(mat)
    # Assert
    assert 0.0 <= out["W"] <= 1.0


def test_perfect_agreement_returns_significant_pvalue():
    # Arrange
    mat = np.tile(np.arange(12).reshape(-1, 1), (1, 5)).astype(float)
    # Act
    out = _w(mat)
    # Assert
    assert out["pvalue"] < 0.01


def test_kendalls_w_returns_required_keys():
    # Arrange
    mat = np.arange(20.0).reshape(5, 4)
    # Act
    out = _w(mat)
    # Assert
    for key in ("W", "S", "n", "k", "dof", "chi2", "pvalue",
                "formatted", "interpretation"):
        assert key in out


def test_use_abs_flag_ranks_absolute_value():
    # Arrange — flipping signs should NOT change result when use_abs=True
    rng = np.random.default_rng(7)
    mat = rng.normal(size=(10, 4))
    # Act
    w_raw = _w(mat, use_abs=True)["W"]
    w_flipped = _w(-mat, use_abs=True)["W"]
    # Assert
    assert pytest.approx(w_raw, abs=1e-12) == w_flipped


def test_returns_dataframe_when_requested():
    # Arrange
    mat = np.arange(20.0).reshape(5, 4)
    # Act
    out = _w(mat, return_as="dataframe")
    # Assert
    assert isinstance(out, pd.DataFrame)


def test_one_dim_input_raises_value_error():
    # Arrange
    bad = np.arange(10.0)
    # Act
    # (call performed inside the pytest.raises context below)
    # Assert
    with pytest.raises(ValueError):
        _w(bad)


def test_interpret_kendalls_w_below_point_one_is_negligible():
    # Arrange
    w = 0.05
    # Act
    label = interpret_kendalls_w(w)
    # Assert
    assert label == "negligible"


def test_interpret_kendalls_w_below_point_three_is_weak():
    # Arrange
    w = 0.2
    # Act
    label = interpret_kendalls_w(w)
    # Assert
    assert label == "weak"


def test_interpret_kendalls_w_below_point_five_is_moderate():
    # Arrange
    w = 0.4
    # Act
    label = interpret_kendalls_w(w)
    # Assert
    assert label == "moderate"


def test_interpret_kendalls_w_below_point_seven_is_strong():
    # Arrange
    w = 0.6
    # Act
    label = interpret_kendalls_w(w)
    # Assert
    assert label == "strong"


def test_interpret_kendalls_w_above_point_seven_is_very_strong():
    # Arrange
    w = 0.8
    # Act
    label = interpret_kendalls_w(w)
    # Assert
    assert label == "very strong"


def test_long_format_dataframe_pivots_correctly():
    # Arrange
    long_df = pd.DataFrame({
        "subj": ["a", "a", "b", "b", "c", "c"],
        "rater": ["r1", "r2", "r1", "r2", "r1", "r2"],
        "score": [1.0, 1.5, 2.0, 1.9, 3.0, 3.2],
    })
    # Act
    out = _w(long_df, subj_col="subj", rater_col="rater",
             score_col="score")
    # Assert
    assert out["n"] == 3 and out["k"] == 2
