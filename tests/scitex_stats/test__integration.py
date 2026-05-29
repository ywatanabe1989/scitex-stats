#!/usr/bin/env python3
# File: tests/scitex_stats/test__integration.py

"""Tests for scitex_stats._integration (public integration aggregator).

Mirrors ``src/scitex_stats/_integration.py``. Verifies that the eight
scitex-specific integration names are exposed both from the aggregator
module and from the top-level ``scitex_stats`` package, and exercises
the core adapters:

- ``Stats`` dataclass (always importable),
- ``test_result_to_stats`` (bundle glue; needs scitex-io),
- ``to_figrecipe`` (figrecipe glue; needs figrecipe).

Each test covers exactly one observable contract.
"""

from __future__ import annotations

import pytest

from scitex_stats import _integration

_PUBLIC_NAMES = (
    "Stats",
    "BUNDLE_AVAILABLE",
    "test_result_to_stats",
    "save_stats",
    "load_stats",
    "to_figrecipe",
    "annotate",
    "load_and_annotate",
)


def _figrecipe_available() -> bool:
    """Whether figrecipe's stats-annotation hooks are importable."""
    from scitex_stats import _figrecipe_integration

    return bool(_figrecipe_integration._AVAILABLE)


def _flat_result():
    """A legacy flat-format run_test-shaped result dict."""
    return {
        "name": "control vs treatment",
        "method": "t-test",
        "statistic": 2.5,
        "statistic_name": "t",
        "p_value": 0.003,
        "effect_size": 1.21,
        "ci95": [0.5, 1.8],
    }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
def test_aggregator_exports_all_eight_names():
    # Arrange
    expected = set(_PUBLIC_NAMES)

    # Act
    actual = set(_integration.__all__)

    # Assert
    assert actual == expected


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_aggregator_module_has_name(name):
    # Arrange
    module = _integration

    # Act
    has = hasattr(module, name)

    # Assert
    assert has


@pytest.mark.parametrize("name", _PUBLIC_NAMES)
def test_package_top_level_exposes_name(name):
    # Arrange
    import scitex_stats

    # Act
    has = hasattr(scitex_stats, name)

    # Assert
    assert has


def test_bundle_available_is_bool():
    # Arrange
    flag = _integration.BUNDLE_AVAILABLE

    # Act
    is_bool = isinstance(flag, bool)

    # Assert
    assert is_bool


# ---------------------------------------------------------------------------
# Stats dataclass (always importable)
# ---------------------------------------------------------------------------
def test_stats_dataclass_is_constructible_empty():
    # Arrange
    cls = _integration.Stats

    # Act
    stats = cls(analyses=[])

    # Assert
    assert stats.analyses == []


# ---------------------------------------------------------------------------
# Bundle glue — test_result_to_stats (needs scitex-io)
# ---------------------------------------------------------------------------
def test_test_result_to_stats_produces_single_analysis():
    # Arrange
    pytest.importorskip("scitex_io.bundle")
    result = _flat_result()

    # Act
    stats = _integration.test_result_to_stats(result)

    # Assert
    assert len(stats.analyses) == 1


def test_test_result_to_stats_preserves_p_value():
    # Arrange
    pytest.importorskip("scitex_io.bundle")
    result = _flat_result()

    # Act
    ad = _integration.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["results"]["p_value"] == 0.003


# ---------------------------------------------------------------------------
# figrecipe glue — to_figrecipe (needs figrecipe)
# ---------------------------------------------------------------------------
def test_to_figrecipe_returns_comparisons_dict():
    # Arrange
    pytest.importorskip("figrecipe.utils")
    result = _flat_result()

    # Act
    converted = _integration.to_figrecipe(result)

    # Assert
    assert "comparisons" in converted


@pytest.mark.skipif(
    _figrecipe_available(),
    reason="figrecipe installed; degradation path not exercised here",
)
def test_to_figrecipe_raises_informative_error_without_figrecipe():
    # Arrange
    result = _flat_result()

    # Act
    raises_ctx = pytest.raises(ImportError, match="figrecipe")

    # Assert
    with raises_ctx:
        _integration.to_figrecipe(result)


# EOF
