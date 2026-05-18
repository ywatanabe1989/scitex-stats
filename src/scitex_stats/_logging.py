#!/usr/bin/env python3
# File: src/scitex_stats/_logging.py

"""Logging shim — `scitex_logging` is a hard runtime dependency.

This module historically supported a stdlib-`logging` fallback when
`scitex_logging` wasn't installed, but per
`general/05_development_11_dependency-tiers.md` the package is now in
`[project.dependencies]` (a hard dep). Bare ``try/except ImportError``
is forbidden in ``src/``; the plain import below is the canonical form
once a dep is promoted.

`_SCITEX_LOGGING_AVAILABLE` is retained (always ``True``) for backward
compatibility with downstream gates that may still probe it.
"""

from scitex_logging import getLogger

_SCITEX_LOGGING_AVAILABLE = True

__all__ = ["getLogger", "_SCITEX_LOGGING_AVAILABLE"]

# EOF
