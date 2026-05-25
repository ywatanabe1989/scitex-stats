"""Smoke tests: every demo script must execute end-to-end.

Two categories of "demos":

1. Standalone scripts in ``examples/`` — extracted from ``src/`` so the leaf
   package contains only library code. Invoked by file path with cwd set to
   ``tmp_path`` so generated artefacts (.jpg, .csv) land in pytest's tmp area.
2. Production modules in ``src/`` that ship a small ``if __name__ ==
   "__main__"`` demo at the bottom (typically ``_test_<name>.py`` test
   modules and standalone helpers like ``_power.py``, ``_cohens_d.py``).
   Invoked as ``python -m scitex_stats.<dotted.path>``.

Coverage motivation: the demos add ~1.5 K SLOC to the measured surface.
Without this test they sat at 0 % coverage and dragged the headline metric
down to ~35 %. The point of running them in CI is to catch the kind of
silent breakage that landed four of these in a broken state before.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# -------- file-path demos (examples/) ------------------------------------
EXAMPLE_DEMOS = [
    "examples/correct/demo_correct_bonferroni.py",
    "examples/correct/demo_correct_holm.py",
    "examples/correct/demo_correct_sidak.py",
    "examples/tests/categorical/demo_chi2.py",
    "examples/tests/categorical/demo_cochran_q.py",
    "examples/tests/categorical/demo_fisher.py",
    "examples/tests/categorical/demo_mcnemar.py",
    "examples/tests/correlation/demo_pearson.py",
    "examples/tests/nonparametric/demo_brunner_munzel.py",
    "examples/tests/nonparametric/demo_friedman.py",
    "examples/tests/normality/demo_shapiro.py",
    "examples/tests/parametric/demo_anova.py",
    "examples/tests/parametric/demo_anova_2way.py",
    "examples/tests/parametric/demo_anova_rm.py",
]

# -------- module-path demos (src/ embedded __main__) ---------------------
MODULE_DEMOS = [
    # ``_correct_fdr.py`` is a production module that also ships a __main__
    # demo (rather than a sibling ``_demo_*.py``).
    "scitex_stats.correct._correct_fdr",
    # ``_test_*.py`` modules embed their own demo via ``if __name__ ==
    # '__main__'``. Smoke-running them exercises the same paths a
    # ``_demo_*.py`` would.
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


@pytest.mark.parametrize(
    "rel_path", EXAMPLE_DEMOS, ids=lambda p: Path(p).stem
)
def test_example_demo_runs(rel_path, tmp_path):
    """Each ``examples/**/demo_*.py`` script must run to exit 0."""
    script = REPO_ROOT / rel_path
    assert script.is_file(), f"missing demo script: {script}"
    subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=True,
        timeout=180,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "module", MODULE_DEMOS, ids=lambda m: m.rsplit(".", 1)[-1]
)
def test_module_demo_runs(module, tmp_path):
    """Each ``src/scitex_stats/.../_test_*.py`` or standalone module with
    an embedded ``__main__`` demo must run to exit 0."""
    subprocess.run(
        [sys.executable, "-m", module],
        cwd=tmp_path,
        check=True,
        timeout=180,
        capture_output=True,
    )
