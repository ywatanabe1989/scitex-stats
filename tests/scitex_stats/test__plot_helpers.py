"""Tests for `scitex_stats._plot_helpers`.

Covers the panel composition + figure-bootstrapping utilities that were
previously only reachable through the in-package test demos. Direct
tests keep them honest when the demo path changes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scitex_stats import _plot_helpers as ph

pytest.importorskip("scitex")
pytest.importorskip("figrecipe")


# -------------------- ensure_figure --------------------


def test_ensure_figure_returns_passthrough_when_ax_provided_created():
    # Arrange
    fig, ax = plt.subplots()
    # Act
    created, returned = ph.ensure_figure(plot=True, ax=ax)
    # Assert
    assert created is False
    plt.close(fig)

def test_ensure_figure_returns_passthrough_when_ax_provided_returned():
    # Arrange
    fig, ax = plt.subplots()
    # Act
    created, returned = ph.ensure_figure(plot=True, ax=ax)
    # Assert
    assert returned is ax
    plt.close(fig)


def test_ensure_figure_creates_single_axes_when_plot_true_and_no_ax_created():
    # Arrange
    # Act
    created, ax = ph.ensure_figure(plot=True, ax=None)
    # Assert
    assert created is True
    plt.close("all")

def test_ensure_figure_creates_single_axes_when_plot_true_and_no_ax_case_2():
    # Arrange
    # Act
    created, ax = ph.ensure_figure(plot=True, ax=None)
    # Assert
    assert ax is not None
    plt.close("all")


def test_ensure_figure_creates_ncol_layout_created():
    # Arrange
    # Act
    created, axes = ph.ensure_figure(plot=True, ax=None, ncols=3)
    # Assert
    assert created is True
    plt.close("all")

def test_ensure_figure_creates_ncol_layout_hasattr_axes_len():
    # Arrange
    # Act
    created, axes = ph.ensure_figure(plot=True, ax=None, ncols=3)
    # Assert
    assert hasattr(axes, "__len__") and len(axes) == 3
    plt.close("all")


def test_ensure_figure_honours_figsize_created():
    # Arrange
    # Act
    created, ax = ph.ensure_figure(plot=True, ax=None, figsize=(4, 3))
    # Assert
    assert created is True
    fig = ax.figure if hasattr(ax, "figure") else ax.get_figure()
    w, h = fig.get_size_inches()
    plt.close("all")

def test_ensure_figure_honours_figsize_round():
    # Arrange
    created, ax = ph.ensure_figure(plot=True, ax=None, figsize=(4, 3))
    fig = ax.figure if hasattr(ax, "figure") else ax.get_figure()
    # Act
    w, h = fig.get_size_inches()
    # Assert
    assert (round(w), round(h)) == (4, 3)
    plt.close("all")


def test_ensure_figure_returns_none_when_plot_false_created():
    # Arrange
    # Act
    created, ax = ph.ensure_figure(plot=False, ax=None)
    # Assert
    assert created is False

def test_ensure_figure_returns_none_when_plot_false_ax():
    # Arrange
    # Act
    created, ax = ph.ensure_figure(plot=False, ax=None)
    # Assert
    assert ax is None


# -------------------- compose_panels --------------------


def _panel_line(ax):
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_title("line")


def _panel_scatter(ax):
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(size=20), rng.normal(size=20))
    ax.set_title("scatter")


def test_compose_panels_horizontal_returns_figure():
    # Arrange
    # Act
    fig = ph.compose_panels([_panel_line, _panel_scatter], layout="horizontal")
    # Assert
    assert fig is not None
    plt.close("all")


def test_compose_panels_vertical_returns_figure():
    # Arrange
    # Act
    fig = ph.compose_panels([_panel_line, _panel_scatter], layout="vertical")
    # Assert
    assert fig is not None
    plt.close("all")


def test_compose_panels_grid_layout_2x2():
    # Arrange
    # Act
    fig = ph.compose_panels(
        [_panel_line, _panel_scatter, _panel_line, _panel_scatter],
        layout=(2, 2),
    )
    # Assert
    assert fig is not None
    plt.close("all")


# -------------------- get_palette --------------------


def test_get_palette_returns_requested_length():
    # Arrange
    # Act
    palette = ph.get_palette(4)
    # Assert
    assert len(palette) == 4


# -------------------- stats_text_box --------------------


def test_stats_text_box_writes_text_artist():
    # Arrange
    fig, ax = plt.subplots()
    # Act
    ph.stats_text_box(ax, ["mean = 1.23", "n = 30"])
    text_artists = [t for t in ax.texts]
    # Assert
    assert text_artists, "stats_text_box should leave at least one Text artist"
    plt.close(fig)


# -------------------- significance_bracket --------------------


def test_significance_bracket_attaches_line_and_text_n_before_lines_lines_ax():
    # Arrange
    fig, ax = plt.subplots()
    g1 = np.array([1.0, 2.0, 3.0])
    g2 = np.array([4.0, 5.0, 6.0])
    n_before_lines = len(ax.lines)
    n_before_text = len(ax.texts)
    # Act
    ph.significance_bracket(ax, 0, 1, "*", [g1, g2])
    # Assert
    assert len(ax.lines) > n_before_lines
    plt.close(fig)

def test_significance_bracket_attaches_line_and_text_n_before_text_texts_ax():
    # Arrange
    fig, ax = plt.subplots()
    g1 = np.array([1.0, 2.0, 3.0])
    g2 = np.array([4.0, 5.0, 6.0])
    n_before_lines = len(ax.lines)
    n_before_text = len(ax.texts)
    # Act
    ph.significance_bracket(ax, 0, 1, "*", [g1, g2])
    # Assert
    assert len(ax.texts) > n_before_text
    plt.close(fig)
