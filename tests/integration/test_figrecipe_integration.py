"""Integration test for the scitex_stats → figrecipe bridge.

`_figrecipe_integration.to_figrecipe / annotate / load_and_annotate`
are thin wrappers around figrecipe's `from_scitex_stats`,
`annotate_from_stats`, and `load_stats_bundle`. The wrappers add:
- ImportError guards when figrecipe isn't installed,
- DataFrame/dict/list input handling,
- AxisWrapper unwrapping (`_axis_mpl` / `_ax`).

This exercises each entry point end-to-end with figrecipe installed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from scitex_stats import _figrecipe_integration as fri  # noqa: E402

figrecipe = pytest.importorskip("figrecipe")


def _stats_result():
    """A run_test-shaped result dict for two-group comparison.

    Note: figrecipe's `from_scitex_stats` looks at ``p_value`` (snake
    case), so we use that key here. The current bridge module does no
    rekey-translation, so a dict using scitex-stats's native
    ``pvalue`` would be silently dropped — separate interop fix.
    """
    return {
        "name": "control vs treatment",
        "method": "Student's t-test (independent)",
        "p_value": 0.0022,
        "stars": "**",
        "effect_size": -0.83,
    }


def test_to_figrecipe_single_dict():
    out = fri.to_figrecipe(_stats_result())
    assert isinstance(out, dict)
    assert "comparisons" in out


def test_to_figrecipe_list_input():
    out = fri.to_figrecipe([_stats_result(), _stats_result()])
    assert isinstance(out, dict)
    assert "comparisons" in out
    assert len(out["comparisons"]) == 2


def test_annotate_attaches_artists_to_axes():
    """`annotate` requires a figrecipe RecordingAxes (has `add_stat_annotation`)."""
    fig, ax = figrecipe.subplots()
    ax.bar(["control", "treatment"], [1.0, 2.0])

    artists = fri.annotate(
        ax,
        _stats_result(),
        positions={"control": 0, "treatment": 1},
        style="stars",
    )
    assert isinstance(artists, list)
    plt.close(fig)


def test_annotate_unwraps_scitex_axis_wrapper():
    """`annotate` should look through `_axis_mpl` / `_ax` attrs."""
    fig, fr_ax = figrecipe.subplots()
    fr_ax.bar(["control", "treatment"], [1.0, 2.0])

    class _Wrapper:
        _axis_mpl = fr_ax

    artists = fri.annotate(
        _Wrapper(),
        _stats_result(),
        positions={"control": 0, "treatment": 1},
    )
    assert isinstance(artists, list)
    plt.close(fig)


def test_annotate_passes_through_already_converted_dict():
    converted = fri.to_figrecipe(_stats_result())
    fig, ax = figrecipe.subplots()
    ax.bar(["control", "treatment"], [1.0, 2.0])
    artists = fri.annotate(ax, converted, positions={"control": 0, "treatment": 1})
    assert isinstance(artists, list)
    plt.close(fig)


def test_load_and_annotate_round_trip(tmp_path):
    """Persist a stats bundle via figrecipe, then load + annotate from disk."""
    bundle_dir = tmp_path / "stats_bundle"
    bundle_dir.mkdir()

    # figrecipe.save_stats_bundle is the canonical writer paired with
    # load_stats_bundle; fall back to a JSON-ish dict if not present.
    save = getattr(figrecipe, "save_stats_bundle", None)
    if save is None:
        pytest.skip("figrecipe.save_stats_bundle not available in this version")

    converted = fri.to_figrecipe(_stats_result())
    bundle_path = bundle_dir / "stats.statsz"
    save(converted, str(bundle_path))

    fig, ax = plt.subplots()
    ax.bar(["control", "treatment"], [1.0, 2.0])
    artists = fri.load_and_annotate(
        ax, str(bundle_path), positions={"control": 0, "treatment": 1}
    )
    assert isinstance(artists, list)
    plt.close(fig)
