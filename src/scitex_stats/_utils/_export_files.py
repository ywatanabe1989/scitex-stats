#!/usr/bin/env python3
# File: scitex_stats/_utils/_export_files.py
# ----------------------------------------
from __future__ import annotations

"""File-export wrappers and the in-process `convert_results` switch.

Split out from `_normalizers.py` so the file-size budget stays
honoured. Re-exported through `_normalizers.export_results`,
`export_summary`, `export_excel_styled`, `convert_results` for
backward compatibility.
"""

import json as _json
import os
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from ._normalize_core import force_dataframe, to_dict


def export_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    path: str,
    format: Optional[str] = None,
    **kwargs,
) -> str:
    """Export test results to a file, format inferred from extension."""
    if format is None:
        _, ext = os.path.splitext(path)
        format = ext.lstrip(".").lower()

    if not isinstance(results, pd.DataFrame):
        df = force_dataframe(results)
    else:
        df = results

    from ._export_reports import _get_scitex_signature

    if format == "csv":
        df.to_csv(path, index=False, **kwargs)
        with open(path, "a") as f:
            f.write(f"\n# {_get_scitex_signature('excel')}\n")
    elif format in ("txt", "tsv"):
        df.to_csv(path, index=False, sep="\t", **kwargs)
        with open(path, "a") as f:
            f.write(f"\n{_get_scitex_signature('text')}")
    elif format == "json":
        data = {
            "data": df.to_dict("records"),
            "metadata": {
                "generated_by": "SciTeX Stats",
                "timestamp": _get_scitex_signature("excel").split(" | ")[1],
                "description": (
                    "Professional Statistical Analysis Framework "
                    "for Scientific Computing"
                ),
            },
        }
        with open(path, "w") as f:
            _json.dump(data, f, indent=2, **kwargs)
    elif format == "xlsx":
        df.to_excel(path, index=False, **kwargs)
    elif format in ("latex", "tex"):
        latex_str = df.to_latex(index=False, **kwargs)
        latex_str += f"\n% {_get_scitex_signature('excel')}\n"
        with open(path, "w") as f:
            f.write(latex_str)
    else:
        raise ValueError(
            f"Unsupported format: {format}. "
            "Use 'csv', 'txt', 'json', 'xlsx', or 'latex'"
        )
    return path


def export_excel_styled(
    results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    path: str,
    **kwargs,
) -> str:
    """Reserved for future styled Excel output (currently a no-op)."""
    return path


def export_summary(
    results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    path: str,
    columns: Optional[List[str]] = None,
    format: Optional[str] = None,
    **kwargs,
) -> str:
    """Export a column-filtered summary of test results."""
    if columns is None:
        columns = [
            "test_method",
            "var_x",
            "var_y",
            "n_x",
            "n_y",
            "statistic",
            "pvalue",
            "pstars",
            "rejected",
            "effect_size",
            "effect_size_metric",
            "effect_size_interpretation",
        ]

    if not isinstance(results, pd.DataFrame):
        df = force_dataframe(results)
    else:
        df = results

    available_cols = [c for c in columns if c in df.columns]
    df_summary = df[available_cols]
    return export_results(df_summary, path, format=format, **kwargs)


def convert_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    return_as: Literal[
        "dict", "dataframe", "markdown", "json", "latex", "html", "text"
    ] = "dict",
    **kwargs,
) -> Union[dict, List[dict], pd.DataFrame, str]:
    """Convert results between in-memory formats (no file I/O)."""
    if return_as == "dict":
        if isinstance(results, dict):
            return results
        if isinstance(results, list):
            return results
        if isinstance(results, pd.DataFrame):
            return (
                results.to_dict("records")
                if len(results) > 1
                else to_dict(results, row=0)
            )

    if return_as == "dataframe":
        return force_dataframe(results)

    if return_as == "markdown":
        return force_dataframe(results).to_markdown(index=False, **kwargs)

    if return_as == "json":
        return force_dataframe(results).to_json(orient="records", indent=2, **kwargs)

    if return_as == "latex":
        return force_dataframe(results).to_latex(index=False, **kwargs)

    if return_as == "html":
        return force_dataframe(results).to_html(index=False, **kwargs)

    if return_as == "text":
        return force_dataframe(results).to_string(index=False, **kwargs)

    if return_as == "csv":
        return force_dataframe(results).to_csv(index=False, **kwargs)

    raise ValueError(
        f"Unknown return_as format: {return_as}. "
        "Use 'dict', 'dataframe', 'markdown', 'json', 'latex', 'html', or 'text'"
    )


# EOF
