"""Tests for `scitex_stats.effect_sizes._effect_size_from_ci`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scitex_stats.effect_sizes._effect_size_from_ci import (
    effect_size_from_ci,
    interpret_effect_size_from_ci,
)


def test_effect_size_from_ci_matches_hand_derived_se_formula():
    # Arrange
    estimate = 0.5
    ci_lower = 0.3
    ci_upper = 0.7
    z = norm.ppf(0.975)
    expected_se = (ci_upper - ci_lower) / (2.0 * z)
    expected = estimate / expected_se
    # Act
    result = effect_size_from_ci(estimate, ci_lower, ci_upper, ci=95)
    # Assert
    assert np.isclose(result, expected)


def test_effect_size_from_ci_zero_estimate_gives_zero():
    # Arrange
    estimate = 0.0
    ci_lower = -0.2
    ci_upper = 0.2
    # Act
    result = effect_size_from_ci(estimate, ci_lower, ci_upper, ci=95)
    # Assert
    assert np.isclose(result, 0.0)


def test_effect_size_from_ci_narrower_ci_gives_larger_effect_size():
    # Arrange
    estimate = 0.5
    narrow = effect_size_from_ci(estimate, 0.45, 0.55, ci=95)
    # Act
    wide = effect_size_from_ci(estimate, 0.2, 0.8, ci=95)
    # Assert
    assert narrow > wide


def test_effect_size_from_ci_generalizes_to_other_ci_levels():
    # Arrange
    estimate = 0.5
    ci_lower = 0.3
    ci_upper = 0.7
    z90 = norm.ppf(0.5 + 90 / 200.0)
    expected_se_90 = (ci_upper - ci_lower) / (2.0 * z90)
    expected_90 = estimate / expected_se_90
    # Act
    result_90 = effect_size_from_ci(estimate, ci_lower, ci_upper, ci=90)
    # Assert
    assert np.isclose(result_90, expected_90)


def test_effect_size_from_ci_raises_on_reversed_bounds():
    # Arrange
    estimate = 0.5
    ci_lower = 0.7
    ci_upper = 0.3
    # Act
    # (call happens inside the Assert block below)
    # Assert
    with pytest.raises(ValueError):
        effect_size_from_ci(estimate, ci_lower, ci_upper, ci=95)


def test_effect_size_from_ci_raises_on_equal_bounds():
    # Arrange
    estimate = 0.5
    ci_lower = 0.5
    ci_upper = 0.5
    # Act
    # (call happens inside the Assert block below)
    # Assert
    with pytest.raises(ValueError):
        effect_size_from_ci(estimate, ci_lower, ci_upper, ci=95)


def test_interpret_effect_size_from_ci_small():
    # Arrange
    d = 0.3
    # Act
    label = interpret_effect_size_from_ci(d)
    # Assert
    assert label == "small"


def test_interpret_effect_size_from_ci_large():
    # Arrange
    d = 0.9
    # Act
    label = interpret_effect_size_from_ci(d)
    # Assert
    assert label == "large"
