#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex_stats/_utils/_normalizers.py
# ----------------------------------------
from __future__ import annotations

"""Output format normalization utilities for scitex.stats.

Thin orchestrator — the actual implementations live in three
sibling modules, split out so each file stays under the 512-LOC
project budget:

- `_normalize_core.py` — schemas + dict/DataFrame coercion.
- `_export_files.py`   — `export_results`, `export_summary`,
                          `export_excel_styled`, `convert_results`.
- `_export_reports.py` — `export_report` + HTML / MD / text helpers.

Existing `from scitex_stats._utils._normalizers import …` callsites
keep working through the re-exports below.
"""

from ._export_files import (
    convert_results,
    export_excel_styled,
    export_results,
    export_summary,
)
from ._export_reports import export_report
from ._normalize_core import (
    COLUMN_TYPES,
    STANDARD_COLUMNS,
    STANDARD_DEFAULTS,
    combine_results,
    force_dataframe,
    normalize_result,
    to_dataframe,
    to_dict,
)

# Convenience alias kept for backward compatibility.
as_dataframe = force_dataframe

__all__ = [
    "normalize_result",
    "to_dataframe",
    "force_dataframe",
    "to_dict",
    "combine_results",
    "convert_results",
    "export_results",
    "export_summary",
    "export_excel_styled",
    "export_report",
    "as_dataframe",
    "STANDARD_COLUMNS",
    "STANDARD_DEFAULTS",
    "COLUMN_TYPES",
]

# EOF
