#!/usr/bin/env python3
# File: scitex_stats/tests/agreement/__init__.py

"""
Inter-rater agreement tests.

Tests for measuring agreement among multiple raters who scored or ranked
the same set of subjects.

Available Tests
---------------
Rank concordance:
    test_kendalls_w : Kendall's coefficient of concordance W (1939)
                      ∈ [0, 1]; agreement on the *ordering* of items
                      across k raters.

Intraclass correlation:
    test_icc : ICC (Shrout & Fleiss 1979).
               Reports ICC(1,1), ICC(2,1), ICC(3,1) (single measures)
               and ICC(1,k), ICC(2,k), ICC(3,k) (average measures).
"""

from ._test_icc import test_icc
from ._test_kendalls_w import test_kendalls_w

__all__ = ["test_kendalls_w", "test_icc"]
