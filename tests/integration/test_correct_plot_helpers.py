"""Cover the `_plot_*` helpers in `scitex_stats.correct`.

The public `correct_*` functions all have a `plot=True` branch that
calls a sibling `_plot_*` helper. The existing unit tests exercise
the maths-only path, so the plot helpers (~30 LOC each) went
uncovered.

Lives in `tests/integration/` because the plot helpers are spread
across four sibling source files; the audit's PS-204 mirror rule
expects per-source test files, but here a single test file naturally
groups the cross-file plot-path concern.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from scitex_stats.correct import (
    correct_bonferroni,
    correct_fdr,
    correct_holm,
    correct_sidak,
)

_RESULTS = [
    {"var_x": "A", "var_y": "B", "pvalue": 0.001},
    {"var_x": "A", "var_y": "C", "pvalue": 0.010},
    {"var_x": "A", "var_y": "D", "pvalue": 0.050},
    {"var_x": "B", "var_y": "C", "pvalue": 0.100},
    {"var_x": "B", "var_y": "D", "pvalue": 0.200},
]


@pytest.mark.parametrize(
    "fn",
    [correct_bonferroni, correct_fdr, correct_holm, correct_sidak],
    ids=lambda f: f.__name__,
)
def test_correct_with_plot_creates_axes(fn):
    """Driving `plot=True` exercises the sibling `_plot_*` helper."""
    # Arrange
    fig, ax = plt.subplots()
    # Act
    fn(_RESULTS.copy(), alpha=0.05, plot=True, ax=ax, verbose=False)
    # Assert
    assert ax.get_title()
    plt.close(fig)


@pytest.mark.parametrize(
    "fn",
    [correct_bonferroni, correct_fdr, correct_holm, correct_sidak],
    ids=lambda f: f.__name__,
)
def test_correct_with_plot_no_ax_creates_figure(fn):
    """Without `ax=`, the helper should create its own figure."""
    # Arrange
    # Act
    fn(_RESULTS.copy(), alpha=0.05, plot=True, verbose=False)
    # Assert
    assert plt.get_fignums(), f"{fn.__name__} did not create a figure"
    plt.close("all")
