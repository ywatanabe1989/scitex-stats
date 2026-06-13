"""Tests for ``scitex_stats.power._power`` (power analysis)."""

from __future__ import annotations

import pytest

from scitex_stats.power._power import power_ttest, sample_size_ttest

# ----- power_ttest --------------------------------------------------------- #


def test_power_in_unit_interval_two_sample():
    # Arrange
    effect_size, n1, n2 = 0.5, 30, 30
    # Act
    p = power_ttest(effect_size=effect_size, n1=n1, n2=n2)
    # Assert
    assert 0.0 <= p <= 1.0


def test_power_grows_with_sample_size():
    """Larger n with fixed d → higher power (monotonic)."""
    # Arrange
    effect_size = 0.5
    # Act
    p_small = power_ttest(effect_size=effect_size, n1=10, n2=10)
    p_med = power_ttest(effect_size=effect_size, n1=30, n2=30)
    p_big = power_ttest(effect_size=effect_size, n1=100, n2=100)
    # Assert
    assert p_small < p_med < p_big


def test_power_grows_with_effect_size():
    """Larger d with fixed n → higher power."""
    # Arrange
    n1, n2 = 50, 50
    # Act
    p_tiny = power_ttest(effect_size=0.1, n1=n1, n2=n2)
    p_med = power_ttest(effect_size=0.5, n1=n1, n2=n2)
    p_huge = power_ttest(effect_size=1.0, n1=n1, n2=n2)
    # Assert
    assert p_tiny < p_med < p_huge


def test_power_zero_effect_equals_alpha_under_two_sided():
    """No real effect: power should be ≈ alpha (probability of false positive)."""
    # Arrange
    alpha = 0.05
    # Act
    p = power_ttest(effect_size=0.0, n1=30, n2=30, alpha=alpha)
    # Assert
    assert abs(p - alpha) < 0.01


def test_one_sample_test_uses_n_kwarg():
    # Arrange
    effect_size, n = 0.5, 40
    # Act
    p = power_ttest(effect_size=effect_size, n=n, test_type="one-sample")
    # Assert
    assert 0.0 <= p <= 1.0


def test_paired_test_uses_n_kwarg():
    # Arrange
    effect_size, n = 0.5, 40
    # Act
    p = power_ttest(effect_size=effect_size, n=n, test_type="paired")
    # Assert
    assert 0.0 <= p <= 1.0


def test_alpha_lowering_reduces_power():
    """Stricter alpha → harder to reject → lower power for same effect/n."""
    # Arrange
    effect_size, n1, n2 = 0.5, 30, 30
    # Act
    p_loose = power_ttest(effect_size=effect_size, n1=n1, n2=n2, alpha=0.05)
    p_strict = power_ttest(effect_size=effect_size, n1=n1, n2=n2, alpha=0.001)
    # Assert
    assert p_strict < p_loose


# ----- sample_size_ttest --------------------------------------------------- #


def test_sample_size_two_sample_returns_int_or_tuple():
    """Two-sample sample_size_ttest returns either an int (equal n) or a (n1, n2) tuple."""
    # Arrange
    effect_size, power = 0.5, 0.80
    # Act
    n = sample_size_ttest(effect_size=effect_size, power=power)
    # Assert
    assert isinstance(n, (int, tuple))


def test_sample_size_two_sample_components_are_positive():
    """Every element of the two-sample sample_size_ttest result is a positive int."""
    # Arrange
    effect_size, power = 0.5, 0.80
    n = sample_size_ttest(effect_size=effect_size, power=power)
    components = n if isinstance(n, tuple) else (n,)
    # Act
    all_positive_ints = all(isinstance(v, int) and v > 0 for v in components)
    # Assert
    assert all_positive_ints


def test_sample_size_decreases_with_larger_effect():
    """Bigger d → fewer samples needed for same power."""
    # Arrange
    n_small_d = sample_size_ttest(effect_size=0.2, power=0.80)
    n_big_d = sample_size_ttest(effect_size=0.8, power=0.80)
    # Act
    a = n_small_d if isinstance(n_small_d, int) else n_small_d[0]
    b = n_big_d if isinstance(n_big_d, int) else n_big_d[0]
    # Assert
    assert a > b


def test_sample_size_increases_with_higher_target_power():
    # Arrange
    n_low = sample_size_ttest(effect_size=0.5, power=0.50)
    n_high = sample_size_ttest(effect_size=0.5, power=0.95)
    # Act
    a = n_low if isinstance(n_low, int) else n_low[0]
    b = n_high if isinstance(n_high, int) else n_high[0]
    # Assert
    assert a < b


def test_sample_size_increases_with_stricter_alpha():
    # Arrange
    n_loose = sample_size_ttest(effect_size=0.5, power=0.80, alpha=0.05)
    n_strict = sample_size_ttest(effect_size=0.5, power=0.80, alpha=0.001)
    # Act
    a = n_loose if isinstance(n_loose, int) else n_loose[0]
    b = n_strict if isinstance(n_strict, int) else n_strict[0]
    # Assert
    assert a < b


def test_one_sample_returns_int():
    # Arrange
    effect_size, power = 0.5, 0.80
    # Act
    n = sample_size_ttest(effect_size=effect_size, power=power, test_type="one-sample")
    # Assert
    assert isinstance(n, int) and n > 0


def test_paired_returns_int():
    # Arrange
    effect_size, power = 0.5, 0.80
    # Act
    n = sample_size_ttest(effect_size=effect_size, power=power, test_type="paired")
    # Assert
    assert isinstance(n, int) and n > 0


def test_round_trip_sample_size_then_power():
    """sample_size_ttest gives n that achieves at least the requested power."""
    # Arrange
    target_power = 0.80
    n = sample_size_ttest(effect_size=0.5, power=target_power)
    n1, n2 = (n, n) if isinstance(n, int) else n
    # Act
    achieved = power_ttest(effect_size=0.5, n1=n1, n2=n2)
    # Assert
    assert achieved >= target_power - 0.02  # Small tolerance for rounding.


@pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
def test_alternative_kwarg_accepted(alt):
    """All three alternative-hypothesis modes should produce a valid power."""
    # Arrange
    effect_size, n1, n2 = 0.5, 30, 30
    # Act
    p = power_ttest(effect_size=effect_size, n1=n1, n2=n2, alternative=alt)
    # Assert
    assert 0.0 <= p <= 1.0
