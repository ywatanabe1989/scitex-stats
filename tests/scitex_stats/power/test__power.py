"""Tests for ``scitex_stats.power._power`` (power analysis)."""

from __future__ import annotations

import pytest

from scitex_stats.power._power import power_ttest, sample_size_ttest

# ----- power_ttest --------------------------------------------------------- #


def test_power_in_unit_interval_two_sample():
    p = power_ttest(effect_size=0.5, n1=30, n2=30)
    assert 0.0 <= p <= 1.0


def test_power_grows_with_sample_size():
    """Larger n with fixed d → higher power (monotonic)."""
    p_small = power_ttest(effect_size=0.5, n1=10, n2=10)
    p_med = power_ttest(effect_size=0.5, n1=30, n2=30)
    p_big = power_ttest(effect_size=0.5, n1=100, n2=100)
    assert p_small < p_med < p_big


def test_power_grows_with_effect_size():
    """Larger d with fixed n → higher power."""
    p_tiny = power_ttest(effect_size=0.1, n1=50, n2=50)
    p_med = power_ttest(effect_size=0.5, n1=50, n2=50)
    p_huge = power_ttest(effect_size=1.0, n1=50, n2=50)
    assert p_tiny < p_med < p_huge


def test_power_zero_effect_equals_alpha_under_two_sided():
    """No real effect: power should be ≈ alpha (probability of false positive)."""
    p = power_ttest(effect_size=0.0, n1=30, n2=30, alpha=0.05)
    assert abs(p - 0.05) < 0.01


def test_one_sample_test_uses_n_kwarg():
    p = power_ttest(effect_size=0.5, n=40, test_type="one-sample")
    assert 0.0 <= p <= 1.0


def test_paired_test_uses_n_kwarg():
    p = power_ttest(effect_size=0.5, n=40, test_type="paired")
    assert 0.0 <= p <= 1.0


def test_alpha_lowering_reduces_power():
    """Stricter alpha → harder to reject → lower power for same effect/n."""
    p_loose = power_ttest(effect_size=0.5, n1=30, n2=30, alpha=0.05)
    p_strict = power_ttest(effect_size=0.5, n1=30, n2=30, alpha=0.001)
    assert p_strict < p_loose


# ----- sample_size_ttest --------------------------------------------------- #


def test_sample_size_returns_positive_int_two_sample():
    n = sample_size_ttest(effect_size=0.5, power=0.80)
    # Two-sample returns either an int (equal n) or tuple (n1, n2).
    if isinstance(n, tuple):
        assert all(isinstance(v, int) and v > 0 for v in n)
    else:
        assert isinstance(n, int) and n > 0


def test_sample_size_decreases_with_larger_effect():
    """Bigger d → fewer samples needed for same power."""
    n_small_d = sample_size_ttest(effect_size=0.2, power=0.80)
    n_big_d = sample_size_ttest(effect_size=0.8, power=0.80)
    a = n_small_d if isinstance(n_small_d, int) else n_small_d[0]
    b = n_big_d if isinstance(n_big_d, int) else n_big_d[0]
    assert a > b


def test_sample_size_increases_with_higher_target_power():
    n_low = sample_size_ttest(effect_size=0.5, power=0.50)
    n_high = sample_size_ttest(effect_size=0.5, power=0.95)
    a = n_low if isinstance(n_low, int) else n_low[0]
    b = n_high if isinstance(n_high, int) else n_high[0]
    assert a < b


def test_sample_size_increases_with_stricter_alpha():
    n_loose = sample_size_ttest(effect_size=0.5, power=0.80, alpha=0.05)
    n_strict = sample_size_ttest(effect_size=0.5, power=0.80, alpha=0.001)
    a = n_loose if isinstance(n_loose, int) else n_loose[0]
    b = n_strict if isinstance(n_strict, int) else n_strict[0]
    assert a < b


def test_one_sample_returns_int():
    n = sample_size_ttest(effect_size=0.5, power=0.80, test_type="one-sample")
    assert isinstance(n, int) and n > 0


def test_paired_returns_int():
    n = sample_size_ttest(effect_size=0.5, power=0.80, test_type="paired")
    assert isinstance(n, int) and n > 0


def test_round_trip_sample_size_then_power():
    """sample_size_ttest gives n that achieves at least the requested power."""
    target_power = 0.80
    n = sample_size_ttest(effect_size=0.5, power=target_power)
    n1, n2 = (n, n) if isinstance(n, int) else n
    achieved = power_ttest(effect_size=0.5, n1=n1, n2=n2)
    assert achieved >= target_power - 0.02  # Small tolerance for rounding.


@pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
def test_alternative_kwarg_accepted(alt):
    """All three alternative-hypothesis modes should produce a valid power."""
    p = power_ttest(effect_size=0.5, n1=30, n2=30, alternative=alt)
    assert 0.0 <= p <= 1.0
