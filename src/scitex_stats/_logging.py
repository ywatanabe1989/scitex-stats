#!/usr/bin/env python3
# File: src/scitex_stats/_logging.py

"""Logging compatibility — use scitex.logging when available, else stdlib."""

import logging as _stdlib_logging

try:
    from scitex_logging import getLogger

    _SCITEX_LOGGING_AVAILABLE = True
except ImportError:
    getLogger = _stdlib_logging.getLogger
    _SCITEX_LOGGING_AVAILABLE = False

__all__ = ["getLogger", "_SCITEX_LOGGING_AVAILABLE"]

# EOF
