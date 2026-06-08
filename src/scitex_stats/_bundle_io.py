#!/usr/bin/env python3
# File: src/scitex_stats/_bundle_io.py
"""scitex_stats ↔ SciTeX bundle I/O.

Convert ``run_test``-shaped result dicts into the canonical
``scitex_stats._dataclasses.Stats`` schema, and persist / load them as
SciTeX bundles (``kind="stats"``) via :mod:`scitex_io.bundle`.

``scitex-io`` is an OPTIONAL dependency (extra ``[bundle]``). This module
imports cleanly without it; the three public functions raise a clear
``ImportError`` when called in an environment that lacks ``scitex-io``::

    pip install scitex-stats[bundle]   # or: uv pip install 'scitex-stats[bundle]'

The statistical dataclasses (``Stats``, ``Analysis``, ``StatMethod``,
``StatResult``, ``EffectSize``) live here in ``scitex_stats`` as the single
source of truth; ``scitex_io.bundle`` imports them from this package. Hence
``test_result_to_stats`` only needs scitex-io for API consistency with the
bundle round-trip, not for the conversion itself.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Optional scitex-io.bundle import. The Bundle class is the only piece that
# genuinely requires scitex-io; the Stats schema itself is local (below).
# ---------------------------------------------------------------------------
try:
    from scitex_io.bundle import Bundle as _Bundle  # type: ignore[import-not-found]

    BUNDLE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via sys.modules shim test
    _Bundle = None  # type: ignore[assignment]
    BUNDLE_AVAILABLE = False

_MISSING_MSG = (
    "scitex-io is required for stats bundle I/O; install scitex-stats[bundle]"
)


def _require_bundle() -> None:
    if not BUNDLE_AVAILABLE:
        raise ImportError(_MISSING_MSG)


def test_result_to_stats(result: Dict[str, Any]):
    """Convert a test-result dict to the canonical ``Stats`` schema.

    Parameters
    ----------
    result : dict
        Test result dictionary. Supports both formats:

        Legacy flat format::

            {"name": "Control vs Treatment", "method": "t-test",
             "p_value": 0.003, "effect_size": 1.21, "ci95": [0.5, 1.8]}

        New nested format (from test functions)::

            {"method": {"name": "t-test", "variant": "independent"},
             "results": {"statistic": 2.5, "statistic_name": "t",
                         "p_value": 0.01}}

    Returns
    -------
    Stats
        Stats object suitable for bundle storage.

    Raises
    ------
    ImportError
        If scitex-io is not installed.
    """
    _require_bundle()

    from scitex_stats._dataclasses import (
        Analysis,
        EffectSize,
        StatMethod,
        StatResult,
        Stats,
    )

    method_data = result.get("method", {})
    if isinstance(method_data, str):
        method = StatMethod(name=method_data, variant=None, parameters={})
        effect_size = None
        es_val = result.get("effect_size")
        if es_val is not None:
            ci = result.get("ci95", [])
            effect_size = EffectSize(
                name="d",
                value=float(es_val),
                ci_lower=ci[0] if len(ci) > 0 else None,
                ci_upper=ci[1] if len(ci) > 1 else None,
            )
        stat_result = StatResult(
            statistic=result.get("statistic", 0.0),
            statistic_name=result.get("statistic_name", ""),
            p_value=result.get("p_value", 1.0),
            df=result.get("df"),
            effect_size=effect_size,
            significant=result.get("p_value", 1.0) < 0.05,
            alpha=0.05,
        )
        analysis_name = result.get("name", "comparison")
    else:
        method = StatMethod(
            name=method_data.get("name", "unknown"),
            variant=method_data.get("variant"),
            parameters=method_data.get("parameters", {}),
        )
        results_data = result.get("results", {})
        effect_size = None
        if "effect_size" in results_data:
            es = results_data["effect_size"]
            effect_size = EffectSize(
                name=es.get("name", ""),
                value=es.get("value", 0.0),
                ci_lower=es.get("ci_lower"),
                ci_upper=es.get("ci_upper"),
            )
        stat_result = StatResult(
            statistic=results_data.get("statistic", 0.0),
            statistic_name=results_data.get("statistic_name", ""),
            p_value=results_data.get("p_value", 1.0),
            df=results_data.get("df"),
            effect_size=effect_size,
            significant=results_data.get("significant"),
            alpha=results_data.get("alpha", 0.05),
        )
        analysis_name = result.get("name", "analysis")

    inputs = dict(result.get("inputs", {}))
    inputs["comparison_name"] = analysis_name
    analysis = Analysis(
        result_id=str(uuid.uuid4()),
        method=method,
        results=stat_result,
        inputs=inputs,
    )

    return Stats(analyses=[analysis])


def save_stats(
    comparisons: Union[List[Dict[str, Any]], Any],
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    as_zip: bool = False,
) -> Path:
    """Save statistical results as a SciTeX bundle (``kind="stats"``).

    Parameters
    ----------
    comparisons : list of dict, or Stats
        List of comparison-result dicts (each converted via
        :func:`test_result_to_stats`), or an already-built ``Stats`` object.
    path : str or Path
        Output bundle path (directory, or ``.zip`` when ``as_zip``).
    metadata : dict, optional
        Currently unused placeholder kept for API stability.
    as_zip : bool, optional
        If True, save as a ZIP archive (``.zip`` suffix enforced).

    Returns
    -------
    Path
        Path to the saved bundle.

    Raises
    ------
    ImportError
        If scitex-io is not installed.
    """
    _require_bundle()

    from scitex_stats._dataclasses import Stats

    p = Path(path)
    if as_zip and p.suffix != ".zip":
        p = p.with_suffix(".zip")

    bundle = _Bundle(p, create=True, kind="stats")

    if comparisons:
        if isinstance(comparisons, list) and isinstance(comparisons[0], dict):
            stats = Stats(analyses=[])
            for comp in comparisons:
                analysis_stats = test_result_to_stats(comp)
                stats.analyses.extend(analysis_stats.analyses)
            bundle.stats = stats
        else:
            bundle.stats = comparisons

    # Stats bundles carry no renderable figure; validation reads payload from
    # disk before save writes it, so skip both render and validate here.
    bundle.save(validate=False, render=False)
    return p


def load_stats(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a stats bundle into a flat, plot-friendly dict.

    Parameters
    ----------
    path : str or Path
        Path to the bundle.

    Returns
    -------
    dict
        ``{"comparisons": [...], "metadata": {...}}`` where each comparison
        is a flat dict (name, method, p_value, effect_size, ci95, formatted).

    Raises
    ------
    ImportError
        If scitex-io is not installed.
    """
    _require_bundle()

    bundle = _Bundle(path)

    comparisons: List[Dict[str, Any]] = []
    if bundle.stats and bundle.stats.analyses:
        for analysis in bundle.stats.analyses:
            ad = analysis.to_dict()
            p_val = ad.get("results", {}).get("p_value", 1.0)
            es_data = ad.get("results", {}).get("effect_size", {})
            es_val = es_data.get("value", 0.0) if es_data else 0.0
            ci = [es_data.get("ci_lower"), es_data.get("ci_upper")] if es_data else []
            ci = [v for v in ci if v is not None]

            if p_val < 0.001:
                formatted = "***"
            elif p_val < 0.01:
                formatted = "**"
            elif p_val < 0.05:
                formatted = "*"
            else:
                formatted = "ns"

            comparisons.append(
                {
                    "name": ad.get("inputs", {}).get("comparison_name", "comparison"),
                    "method": ad.get("method", {}).get("name", "unknown"),
                    "p_value": p_val,
                    "effect_size": es_val,
                    "ci95": ci,
                    "formatted": formatted,
                }
            )

    return {
        "comparisons": comparisons,
        "metadata": bundle.spec.to_dict() if bundle.spec else {},
    }


__all__ = [
    "BUNDLE_AVAILABLE",
    "test_result_to_stats",
    "save_stats",
    "load_stats",
]

# EOF
