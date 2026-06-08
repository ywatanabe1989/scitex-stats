#!/usr/bin/env python3
# Timestamp: "2026-01-12 (ywatanabe)"
# File: scitex_stats/_figrecipe_integration.py
"""Figrecipe integration for statistical annotations on plots."""

from typing import Any, Dict, List, Optional, Union

from scitex_dev import try_import_optional

# figrecipe is an optional [all]-tier dep; its _integrations submodule is not
# yet in the public API, so a missing module OR attr means "unavailable".
_fr_load_bundle = try_import_optional(
    "figrecipe._integrations._scitex_stats",
    attr="load_stats_bundle",
    extra="all",
    pkg="scitex-stats",
)
_fr_annotate = try_import_optional(
    "figrecipe.utils", attr="annotate_from_stats", extra="all", pkg="scitex-stats"
)
_fr_convert = try_import_optional(
    "figrecipe.utils", attr="from_scitex_stats", extra="all", pkg="scitex-stats"
)
_AVAILABLE = all(x is not None for x in (_fr_load_bundle, _fr_annotate, _fr_convert))


def to_figrecipe(
    stats_result: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Convert scitex.stats result(s) to figrecipe format.

    Parameters
    ----------
    stats_result : dict or list of dict
        Statistical result(s) from scitex.stats functions.

    Returns
    -------
    dict
        Figrecipe-compatible format with 'comparisons' list.
    """
    if not _AVAILABLE:
        raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")
    return _fr_convert(stats_result)


def annotate(
    ax,
    stats: Union[Dict[str, Any], List[Dict[str, Any]]],
    positions: Optional[Dict[str, float]] = None,
    style: str = "stars",
    **kwargs,
) -> List[Any]:
    """Add statistical annotations to a plot.

    Parameters
    ----------
    ax : Axes or AxisWrapper
        The axes to annotate.
    stats : dict or list of dict
        Statistical results (auto-converted if needed).
    positions : dict, optional
        Group name to x position mapping.
    style : str
        'stars', 'p_value', or 'both'.

    Returns
    -------
    list
        Created matplotlib artist objects.
    """
    if not _AVAILABLE:
        raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")

    # Convert if needed
    if isinstance(stats, list) or "comparisons" not in stats:
        stats = _fr_convert(stats)

    # Unwrap scitex.plt.AxisWrapper to its inner figrecipe RecordingAxes,
    # which is what `annotate_from_stats` needs (it calls
    # `add_stat_annotation`, defined on RecordingAxes, not raw mpl.Axes).
    # If `ax` is already a RecordingAxes, pass it straight through —
    # `_ax` would unwrap one level too far and land on raw matplotlib.
    fr_ax = getattr(ax, "_axis_mpl", ax)

    return _fr_annotate(fr_ax, stats, positions=positions, style=style, **kwargs)


def load_and_annotate(
    ax,
    path: str,
    positions: Optional[Dict[str, float]] = None,
    style: str = "stars",
    **kwargs,
) -> List[Any]:
    """Load stats from bundle file and annotate plot.

    Parameters
    ----------
    ax : Axes or AxisWrapper
        The axes to annotate.
    path : str
        Path to .statsz or .zip bundle.
    positions : dict, optional
        Group name to x position mapping.
    style : str
        'stars', 'p_value', or 'both'.

    Returns
    -------
    list
        Created matplotlib artist objects.
    """
    if not _AVAILABLE:
        raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")

    fr_stats = _fr_load_bundle(path)
    return annotate(ax, fr_stats, positions=positions, style=style, **kwargs)


__all__ = ["to_figrecipe", "annotate", "load_and_annotate"]

# EOF
