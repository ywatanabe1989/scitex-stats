"""Smoke tests: every `_demo_*.py` script must execute end-to-end.

These demo modules live inside the package and were silently broken
(four out of eleven raised before completion). This parametrised test
runs each one as ``python -m scitex_stats.tests.<cat>._demo_<name>``
in an isolated tmp_path so the suite catches regressions on merge.

Coverage motivation: the demos add ~1.5 K SLOC to the measured surface.
Without this test they sat at 0 % coverage and dragged the headline
metric down to ~35 %.
"""

import subprocess
import sys

import pytest

DEMOS = [
    "scitex_stats.correct._demo_correct_fdr",
    "scitex_stats.tests.categorical._demo_chi2",
    "scitex_stats.tests.categorical._demo_cochran_q",
    "scitex_stats.tests.categorical._demo_fisher",
    "scitex_stats.tests.categorical._demo_mcnemar",
    "scitex_stats.tests.correlation._demo_pearson",
    "scitex_stats.tests.nonparametric._demo_brunner_munzel",
    "scitex_stats.tests.nonparametric._demo_friedman",
    "scitex_stats.tests.normality._demo_shapiro",
    "scitex_stats.tests.parametric._demo_anova",
    "scitex_stats.tests.parametric._demo_anova_2way",
    "scitex_stats.tests.parametric._demo_anova_rm",
]


@pytest.mark.parametrize("module", DEMOS, ids=lambda m: m.rsplit(".", 1)[-1])
def test_demo_runs(module, tmp_path):
    subprocess.run(
        [sys.executable, "-m", module],
        cwd=tmp_path,
        check=True,
        timeout=180,
        capture_output=True,
    )
