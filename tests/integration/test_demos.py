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
    # `_demo_*.py` sibling files (extracted from oversized test modules).
    "scitex_stats.correct._demo_correct_bonferroni",
    "scitex_stats.correct._correct_fdr",  # in-source __main__ demo
    "scitex_stats.correct._demo_correct_holm",
    "scitex_stats.correct._demo_correct_sidak",
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
    # `_test_*.py` modules that embed their own demo via `if __name__
    # == "__main__"`. Smoke-running them exercises the same paths a
    # `_demo_*.py` would. Once a file is split, drop it from this list.
    "scitex_stats.tests.categorical._test_chi2",
    "scitex_stats.tests.categorical._test_cochran_q",
    "scitex_stats.tests.categorical._test_fisher",
    "scitex_stats.tests.correlation._test_kendall",
    "scitex_stats.tests.correlation._test_pearson",
    "scitex_stats.tests.correlation._test_spearman",
    "scitex_stats.tests.correlation._test_theilsen",
    "scitex_stats.tests.nonparametric._test_brunner_munzel",
    "scitex_stats.tests.nonparametric._test_friedman",
    "scitex_stats.tests.nonparametric._test_kruskal",
    "scitex_stats.tests.nonparametric._test_mannwhitneyu",
    "scitex_stats.tests.nonparametric._test_wilcoxon",
    "scitex_stats.tests.normality._test_ks_1samp",
    "scitex_stats.tests.normality._test_ks_2samp",
    "scitex_stats.tests.normality._test_shapiro",
    "scitex_stats.tests.parametric._test_ttest_1samp",
    "scitex_stats.tests.parametric._test_ttest_ind",
    "scitex_stats.tests.parametric._test_ttest_rel",
    "scitex_stats.posthoc._dunnett",
    "scitex_stats.posthoc._games_howell",
    "scitex_stats.posthoc._tukey_hsd",
    # Standalone modules with __main__ demos (effect sizes, power).
    "scitex_stats.effect_sizes._cliffs_delta",
    "scitex_stats.effect_sizes._cohens_d",
    "scitex_stats.effect_sizes._epsilon_squared",
    "scitex_stats.effect_sizes._eta_squared",
    "scitex_stats.effect_sizes._prob_superiority",
    "scitex_stats.power._power",
    "scitex_stats._utils._effect_size",
    "scitex_stats._utils._formatters",
    "scitex_stats._utils._power",
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
