"""Numpy-path tests for `scitex_stats.descriptive._circular`.

The neighbouring `test__circular.py` file skips entirely when torch
isn't installed (`pytest.importorskip("torch")` at module level), so
the numpy backend — the path that actually runs in the lean `[dev]`
environment — went uncovered. These tests exercise it directly so
coverage no longer disappears with torch.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scitex_stats.descriptive import _circular as circ


def _uniform_circle(n_bins: int = 360, batch: int = 1):
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    values = np.ones(n_bins)
    return (
        angles[None, :].repeat(batch, axis=0),
        values[None, :].repeat(batch, axis=0),
    )


def _von_mises_like(mu: float, kappa: float, n_bins: int = 360, batch: int = 1):
    angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    values = np.exp(kappa * np.cos(angles - mu))
    return (
        angles[None, :].repeat(batch, axis=0),
        values[None, :].repeat(batch, axis=0),
    )


# -------------------- circular_mean --------------------


def test_circular_mean_recovers_peak_of_von_mises():
    mu = np.pi / 3
    angles, values = _von_mises_like(mu=mu, kappa=10.0)
    out = circ.circular_mean(angles, values, dim=-1)
    np.testing.assert_allclose(out, [mu], atol=1e-3)


def test_circular_mean_wraps_into_zero_two_pi():
    mu = 2 * np.pi - 0.1
    angles, values = _von_mises_like(mu=mu, kappa=20.0)
    out = circ.circular_mean(angles, values, dim=-1)
    assert np.all((out >= 0) & (out <= 2 * np.pi))
    np.testing.assert_allclose(out, [mu], atol=1e-2)


# -------------------- circular_concentration --------------------


def test_circular_concentration_high_for_peaked():
    angles_p, values_p = _von_mises_like(mu=0.0, kappa=20.0)
    angles_u, values_u = _uniform_circle()
    conc_peaked = circ.circular_concentration(angles_p, values_p, dim=-1)
    conc_uniform = circ.circular_concentration(angles_u, values_u, dim=-1)
    assert conc_peaked[0] > conc_uniform[0]
    assert abs(conc_uniform[0]) < 1e-3
    assert conc_peaked[0] > 0.9


# -------------------- skewness / kurtosis --------------------


def test_circular_skewness_finite():
    angles, values = _von_mises_like(mu=np.pi, kappa=5.0)
    assert np.all(np.isfinite(circ.circular_skewness(angles, values, dim=-1)))


def test_circular_kurtosis_finite():
    angles, values = _von_mises_like(mu=np.pi, kappa=5.0)
    assert np.all(np.isfinite(circ.circular_kurtosis(angles, values, dim=-1)))


# -------------------- describe_circular --------------------


def test_describe_circular_default_funcs():
    angles, values = _von_mises_like(mu=0.5, kappa=8.0)
    stack, names = circ.describe_circular(angles, values, dim=-1)
    assert names == [
        "circular_mean",
        "circular_concentration",
        "circular_skewness",
        "circular_kurtosis",
    ]
    assert stack.shape == (1, 4)


def test_describe_circular_all_funcs():
    angles, values = _von_mises_like(mu=0.5, kappa=8.0)
    stack, names = circ.describe_circular(angles, values, dim=-1, funcs="all")
    assert len(names) == 4
    assert stack.shape == (1, 4)


def test_describe_circular_subset_funcs():
    angles, values = _von_mises_like(mu=0.5, kappa=8.0)
    stack, names = circ.describe_circular(
        angles,
        values,
        dim=-1,
        funcs=["circular_mean", "circular_concentration"],
    )
    assert names == ["circular_mean", "circular_concentration"]
    assert stack.shape == (1, 2)


# -------------------- guard rails --------------------


def test_circular_mean_warns_on_degrees_input():
    angles, values = _von_mises_like(mu=1.0, kappa=2.0)
    angles_deg = angles * (180.0 / np.pi)  # max > 2π → should warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        circ.circular_mean(angles_deg, values, dim=-1)
        assert any("Maximum angle value" in str(item.message) for item in w)


def test_circular_mean_asserts_on_1d_input():
    angles = np.linspace(0, 2 * np.pi, 100)
    values = np.ones(100)
    with pytest.raises(AssertionError, match="at least 2 dimensional"):
        circ.circular_mean(angles, values, dim=-1)


def test_circular_mean_asserts_on_shape_mismatch():
    angles, _ = _von_mises_like(mu=0.0, kappa=5.0, n_bins=360)
    values = np.ones((1, 180))
    with pytest.raises(AssertionError, match="shape"):
        circ.circular_mean(angles, values, dim=-1)
