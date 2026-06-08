#!/usr/bin/env python3
# File: src/scitex_stats/_integration.py
"""SciTeX-specific stats integration (public aggregator).

Single import surface for the two ecosystem adapters that scitex_stats owns:

- **Bundle I/O** (``scitex_io.bundle``): ``Stats``, ``test_result_to_stats``,
  ``save_stats``, ``load_stats``, ``BUNDLE_AVAILABLE`` — re-exported from
  :mod:`scitex_stats._bundle_io`. ``scitex-io`` is an OPTIONAL dependency
  (extra ``[bundle]``); importing this module never requires it, and the
  bundle functions raise a clear ``ImportError`` when called without it.

- **figrecipe** (statistical annotations on plots): ``annotate``,
  ``to_figrecipe``, ``load_and_annotate`` — re-exported from
  :mod:`scitex_stats._figrecipe_integration`. ``figrecipe`` is an OPTIONAL
  dependency (extra ``[figrecipe]``); those functions degrade to an
  informative ``ImportError`` when figrecipe is absent.

Historically this glue lived in the umbrella ``scitex.stats._integration``;
it now lives here in the owning standalone package. The umbrella keeps a thin
alias.
"""

from __future__ import annotations

# Bundle schema + bundle I/O glue (scitex-io optional).
from ._bundle_io import (  # noqa: F401
    BUNDLE_AVAILABLE,
    load_stats,
    save_stats,
    test_result_to_stats,
)

# Bundle schema dataclass (single source of truth, always importable).
from ._dataclasses import Stats  # noqa: F401

# figrecipe annotation glue (figrecipe optional).
from ._figrecipe_integration import (  # noqa: F401
    annotate,
    load_and_annotate,
    to_figrecipe,
)

__all__ = [
    # Bundle schema
    "Stats",
    "BUNDLE_AVAILABLE",
    # Bundle I/O
    "test_result_to_stats",
    "save_stats",
    "load_stats",
    # figrecipe
    "to_figrecipe",
    "annotate",
    "load_and_annotate",
]

# EOF
