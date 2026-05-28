#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Surface contract for ``scitex_stats._dataclasses._Stats``.

The Stats schema dataclasses are the single source of truth across the
ecosystem (umbrella, scitex_io's optional provider, GUI / editor tools).
These tests pin the public exports — a future rename or accidental drop
fails here, not at a downstream import site.

Bundle round-trip semantics + zip handling live in
``tests/scitex_stats/io/test__bundle.py``.
"""

import pytest

import scitex_stats._dataclasses as dc


@pytest.mark.parametrize(
    "name",
    [
        # Version sentinel.
        "STATS_VERSION",
        # Type aliases.
        "PositionMode",
        "UnitType",
        "SymbolStyle",
        # GUI classes.
        "Position",
        "StatStyling",
        "StatPositioning",
        # Core dataclasses.
        "DataRef",
        "EffectSize",
        "StatMethod",
        "StatResult",
        "StatDisplay",
        "Analysis",
        "Stats",
    ],
)
def test_dataclasses_namespace_exposes_symbol(name):
    """Every name declared in ``__all__`` is reachable at the subpackage root."""
    # Arrange
    # Act
    attr = getattr(dc, name, None)
    # Assert
    assert attr is not None


def test_stats_version_is_a_dotted_string():
    """``STATS_VERSION`` follows semver-ish ``major.minor.patch``."""
    # Arrange
    parts = dc.STATS_VERSION.split(".")
    # Act
    is_three_part = len(parts) == 3
    # Assert
    assert is_three_part


def test_stats_class_is_a_dataclass():
    """``Stats`` is exposed as a real dataclass (not a stub or alias)."""
    # Arrange
    import dataclasses

    # Act
    is_dc = dataclasses.is_dataclass(dc.Stats)
    # Assert
    assert is_dc


def test_dataref_dict_round_trip_preserves_path():
    """``DataRef.to_dict`` → ``DataRef.from_dict`` preserves the ``path`` field."""
    # Arrange
    ref = dc.DataRef(path="/data/x.csv", columns=["a", "b"])
    # Act
    rebuilt = dc.DataRef.from_dict(ref.to_dict())
    # Assert
    assert rebuilt.path == "/data/x.csv"


def test_dataref_dict_round_trip_preserves_columns():
    """``DataRef.to_dict`` → ``DataRef.from_dict`` preserves the ``columns`` field."""
    # Arrange
    ref = dc.DataRef(path="/data/x.csv", columns=["a", "b"])
    # Act
    rebuilt = dc.DataRef.from_dict(ref.to_dict())
    # Assert
    assert rebuilt.columns == ["a", "b"]
