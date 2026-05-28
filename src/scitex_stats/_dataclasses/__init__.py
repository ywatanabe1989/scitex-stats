#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistics dataclasses for scitex_stats.

Single source of truth for statistical result schemas across the
scitex ecosystem. Covers both bundle storage (data provenance, test
results) and GUI annotation (positioning, styling for interactive
editing).

Public API mirrors what umbrella callers historically imported from
``scitex.io.bundle.kinds._stats._dataclasses``; that umbrella path now
re-exports from here.
"""

from ._Stats import (  # noqa: F401
    STATS_VERSION,
    Analysis,
    DataRef,
    EffectSize,
    Position,
    PositionMode,
    StatDisplay,
    StatMethod,
    StatPositioning,
    StatResult,
    Stats,
    StatStyling,
    SymbolStyle,
    UnitType,
)

__all__ = [
    "STATS_VERSION",
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    "Position",
    "StatStyling",
    "StatPositioning",
    "DataRef",
    "EffectSize",
    "StatMethod",
    "StatResult",
    "StatDisplay",
    "Analysis",
    "Stats",
]
