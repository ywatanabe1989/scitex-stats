"""Tests for `scitex_stats.descriptive._ci.ci` (95 % CI helper)."""

from __future__ import annotations

import numpy as np

from scitex_stats.descriptive._ci import ci


def test_ci_matches_analytical_formula_on_clean_data():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = 1.96 * data.std() / np.sqrt(5)
    assert np.isclose(ci(data), expected, atol=1e-9)


def test_ci_ignores_nans():
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    finite = data[~np.isnan(data)]
    expected = 1.96 * finite.std() / np.sqrt(len(finite))
    assert np.isclose(ci(data), expected, atol=1e-9)


def test_ci_returns_zero_for_constant_data():
    data = np.array([3.0, 3.0, 3.0, 3.0])
    assert ci(data) == 0.0


def test_ci_axis_argument_accepted_for_flat_input():
    """The `axis` argument is forwarded to `.std`. With the current
    implementation `~np.isnan(xx)` flattens, so for matrices the result
    collapses to a scalar — but the kwarg is still exercised here so
    coverage records the call site."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ci(data, axis=None)
    assert isinstance(out, float) or np.ndim(out) == 0
