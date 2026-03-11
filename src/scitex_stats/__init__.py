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
    _utils,
    auto,
    correct,
    descriptive,
    effect_sizes,
    posthoc,
    power,
    tests,
)
from scitex_stats._dispatch import available_tests, run_test
from scitex_stats._utils._serialize import to_json_safe
from scitex_stats.auto import (
    StatContext,
    StatStyle,
    TestRule,
    check_applicable,
    get_stat_style,
    p_to_stars,
    recommend_tests,
)
from scitex_stats.descriptive import describe
from scitex_stats.tests import (
    test_anova,
    test_anova_2way,
    test_anova_rm,
    test_brunner_munzel,
    test_chi2,
    test_cochran_q,
    test_fisher,
    test_friedman,
    test_kendall,
    test_kruskal,
    test_ks_1samp,
    test_ks_2samp,
    test_mannwhitneyu,
    test_mcnemar,
    test_normality,
    test_pearson,
    test_shapiro,
    test_spearman,
    test_theilsen,
    test_ttest_1samp,
    test_ttest_ind,
    test_ttest_rel,
    test_wilcoxon,
)

__all__ = [
    "__version__",
    # Submodules
    "_utils",
    "auto",
    "correct",
    "descriptive",
    "effect_sizes",
    "posthoc",
    "power",
    "tests",
    # Dispatcher
    "run_test",
    "available_tests",
    # Descriptive
    "describe",
    # JSON serialization
    "to_json_safe",
    # Auto convenience
    "StatContext",
    "TestRule",
    "StatStyle",
    "recommend_tests",
    "check_applicable",
    "get_stat_style",
    "p_to_stars",
    # Parametric (6)
    "test_ttest_ind",
    "test_ttest_rel",
    "test_ttest_1samp",
    "test_anova",
    "test_anova_rm",
    "test_anova_2way",
    # Nonparametric (5)
    "test_brunner_munzel",
    "test_wilcoxon",
    "test_kruskal",
    "test_mannwhitneyu",
    "test_friedman",
    # Correlation (4)
    "test_pearson",
    "test_spearman",
    "test_kendall",
    "test_theilsen",
    # Categorical (4)
    "test_chi2",
    "test_fisher",
    "test_mcnemar",
    "test_cochran_q",
    # Normality (4)
    "test_shapiro",
    "test_normality",
    "test_ks_1samp",
    "test_ks_2samp",
]

# EOF
