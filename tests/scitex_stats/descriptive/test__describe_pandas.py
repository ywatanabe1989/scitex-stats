#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_stats.descriptive.describe_pandas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scitex_stats.descriptive import describe_pandas


@pytest.fixture
def sample_df():
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})


class TestDescribePandasMeanStd:
    def test_returns_n_mean_std(self, sample_df):
        result = describe_pandas(sample_df, method="mean_std")
        assert set(result.keys()) == {"n", "mean", "std"}

    def test_mean_correct(self, sample_df):
        result = describe_pandas(sample_df, method="mean_std")
        np.testing.assert_allclose(result["mean"]["A"], 3.0)
        np.testing.assert_allclose(result["mean"]["B"], 30.0)


class TestDescribePandasMeanCI:
    def test_returns_n_mean_ci(self, sample_df):
        result = describe_pandas(sample_df, method="mean_ci")
        assert set(result.keys()) == {"n", "mean", "ci"}


class TestDescribePandasMedianIQR:
    def test_returns_n_median_iqr(self, sample_df):
        result = describe_pandas(sample_df, method="median_iqr")
        assert set(result.keys()) == {"n", "median", "iqr"}

    def test_median_correct(self, sample_df):
        result = describe_pandas(sample_df, method="median_iqr")
        np.testing.assert_allclose(result["median"]["A"], 3.0)
        np.testing.assert_allclose(result["median"]["B"], 30.0)


class TestDescribePandasInvalid:
    def test_rejects_unknown_method(self, sample_df):
        with pytest.raises(AssertionError):
            describe_pandas(sample_df, method="not_a_method")
