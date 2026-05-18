#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_corrections.py

"""P-value correction handler for multiple comparisons."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = ["correct_pvalues_handler"]


async def correct_pvalues_handler(
    pvalues: list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> dict:
    """Apply multiple comparison correction to p-values."""
    # `statsmodels` is a hard dep (see general/05_development_11) — the
    # historical no-statsmodels fallback was dead code.
    try:
        loop = asyncio.get_event_loop()

        def do_correct():
            from statsmodels.stats.multitest import multipletests

            # Map method names
            method_map = {
                "bonferroni": "bonferroni",
                "fdr_bh": "fdr_bh",
                "fdr_by": "fdr_by",
                "holm": "holm",
                "sidak": "sidak",
            }
            sm_method = method_map.get(method, "fdr_bh")

            pvals = np.array(pvalues)
            reject, pvals_corrected, _, _ = multipletests(
                pvals, alpha=alpha, method=sm_method
            )

            return {
                "original_pvalues": pvalues,
                "corrected_pvalues": pvals_corrected.tolist(),
                "reject_null": reject.tolist(),
                "n_significant": int(reject.sum()),
                "n_tests": len(pvalues),
            }

        result = await loop.run_in_executor(None, do_correct)

        return {
            "success": True,
            "method": method,
            "alpha": alpha,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
