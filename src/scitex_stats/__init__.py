#!/usr/bin/env python3
# File: src/scitex_stats/__init__.py

"""SciTeX Stats - Publication-ready statistical testing framework.

Three Interfaces:
    - Python API: import scitex_stats as ss
    - CLI: scitex-stats <command>
    - MCP: 10 tools for AI agents

Modules:
    - tests: 23 statistical tests (parametric, nonparametric, correlation, categorical, normality)
    - effect_sizes: Cohen's d, Cliff's delta, eta squared, epsilon squared, probability of superiority
    - correct: Multiple comparison corrections (Bonferroni, FDR, Holm, Sidak)
    - posthoc: Post-hoc tests (Tukey HSD, Dunnett, Games-Howell)
    - power: Statistical power analysis and sample size calculation
    - descriptive: Descriptive statistics and confidence intervals
    - auto: Automatic test recommendation
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("scitex-stats")
except _PackageNotFoundError:
    from pathlib import Path as _Path

    _pyproject = _Path(__file__).parent.parent.parent / "pyproject.toml"
    __version__ = "0.0.0"
    if _pyproject.exists():
        with open(_pyproject) as _f:
            for _line in _f:
                if _line.startswith("version"):
                    __version__ = _line.split("=")[1].strip().strip('"')
                    break

# ---------------------------------------------------------------------------
# Core imports — these are the public Python API
# ---------------------------------------------------------------------------

from scitex_stats import (
    auto,
    correct,
    descriptive,
    effect_sizes,
    posthoc,
    power,
    tests,
)
from scitex_stats._dispatch import available_tests, run_test
from scitex_stats.auto import (
    StatContext,
    TestRule,
    p_to_stars,
    recommend_tests,
)
from scitex_stats.descriptive import describe

__all__ = [
    "__version__",
    # Dispatcher
    "run_test",
    "available_tests",
    "describe",
    # Submodules
    "auto",
    "correct",
    "descriptive",
    "effect_sizes",
    "posthoc",
    "power",
    "tests",
    # Auto convenience
    "StatContext",
    "TestRule",
    "recommend_tests",
    "p_to_stars",
]

# EOF
