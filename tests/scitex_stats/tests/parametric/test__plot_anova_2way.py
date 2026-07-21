"""Direct test for `_plot_anova_2way` (was 8 % covered).

The demos exercise this through `test_anova_2way(plot=True)` but only
in subprocess. A direct call with a fabricated results dict pins the
in-process coverage.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scitex_stats.tests.parametric._plot_anova_2way import _plot_anova_2way


def _fake_results(a_levels=("low", "high"), b_levels=("ctrl", "drug")):
    a_n = len(a_levels)
    b_n = len(b_levels)
    cell_means = np.arange(a_n * b_n, dtype=float).reshape(a_n, b_n) + 1.0
    return {
        "cell_means": cell_means,
        "a_levels": list(a_levels),
        "b_levels": list(b_levels),
        "a_marginal_means": cell_means.mean(axis=1).tolist(),
        "b_marginal_means": cell_means.mean(axis=0).tolist(),
        "factor_a_name": "FactorA",
        "factor_b_name": "FactorB",
        "effects": [
            {
                "effect": "FactorA",
                "df_effect": 1,
                "df_error": 20,
                "statistic": 5.0,
                "pvalue": 0.03,
                "stars": "*",
                "effect_size": 0.20,
            },
            {
                "effect": "FactorB",
                "df_effect": 1,
                "df_error": 20,
                "statistic": 9.0,
                "pvalue": 0.006,
                "stars": "**",
                "effect_size": 0.35,
            },
            {
                "effect": "FactorA:FactorB",
                "df_effect": 1,
                "df_error": 20,
                "statistic": 1.0,
                "pvalue": 0.30,
                "stars": "",
                "effect_size": 0.05,
            },
        ],
    }


def test_plot_anova_2way_returns_figure_with_four_panels_fig():
    # Arrange
    # Act
    fig = _plot_anova_2way(_fake_results())
    # Assert
    assert fig is not None
    axes = fig.get_axes()
    plt.close(fig)

def test_plot_anova_2way_returns_figure_with_four_panels_axes():
    # Arrange
    fig = _plot_anova_2way(_fake_results())
    # Act
    axes = fig.get_axes()
    # Assert
    assert len(axes) == 4
    plt.close(fig)


def test_plot_anova_2way_writes_stat_text_box_on_three_panels():
    # Arrange
    fig = _plot_anova_2way(_fake_results())
    axes = fig.get_axes()
    # Act
    text_counts = [len(ax.texts) for ax in axes]
    # Assert
    assert sum(c >= 1 for c in text_counts) == 3
    plt.close(fig)


def test_plot_anova_2way_three_by_two_factor_layout():
    """Non-square design — 3 levels of A, 2 levels of B."""
    # Arrange
    # Act
    fig = _plot_anova_2way(
        _fake_results(
            a_levels=("low", "mid", "high"),
            b_levels=("ctrl", "drug"),
        )
    )
    # Assert
    assert len(fig.get_axes()) == 4
    plt.close(fig)
