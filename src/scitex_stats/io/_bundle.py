#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: scitex_stats/io/_bundle.py

"""SciTeX ``.stats`` bundle I/O — statistics-specific bundle operations.

Two output forms are supported, mirroring figrecipe's ``.plt.zip`` /
``.fig.zip``:

- ``.stats.zip``  — single-file zipped bundle (the recommended portable
  form; consumed by scitex_io's optional provider).
- ``.stats``       — directory bundle (legacy / debugging form, kept for
  iteration on raw files).

Bundle content::

    <bundle>/
        stats.json     # spec (schema, comparisons, descriptive, …)
        data.csv       # supplementary data (optional)
        report.md      # human summary (optional)

Public API::

    save_stats_bundle(data, path)   # path is a dir-like target or a .stats.zip
    load_stats_bundle(path)         # reads either a dir or a .stats.zip
    validate_stats_spec(spec)       # pure validation, no I/O
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Union

__all__ = [
    "STATS_SCHEMA_SPEC",
    "load_stats_bundle",
    "save_stats_bundle",
    "validate_stats_spec",
]

# Schema specification for ``.stats`` bundles.
STATS_SCHEMA_SPEC: Dict[str, Any] = {
    "name": "scitex.stats.stats",
    "version": "1.0.0",
    "required_fields": ["schema"],
    "optional_fields": ["comparisons", "metadata", "descriptive", "test_results"],
}


def validate_stats_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate ``.stats``-specific fields. Returns the list of error messages."""
    errors: List[str] = []

    if "comparisons" in spec:
        comparisons = spec["comparisons"]
        if not isinstance(comparisons, list):
            errors.append("'comparisons' must be a list")
        else:
            for i, comp in enumerate(comparisons):
                if not isinstance(comp, dict):
                    errors.append(f"comparisons[{i}] must be a dictionary")
                    continue

                if "p_value" in comp:
                    p = comp["p_value"]
                    if not isinstance(p, (int, float)):
                        errors.append(f"comparisons[{i}].p_value must be numeric")
                    elif not (0 <= p <= 1):
                        errors.append(
                            f"comparisons[{i}].p_value must be between 0 and 1"
                        )

                if "effect_size" in comp:
                    es = comp["effect_size"]
                    if isinstance(es, dict):
                        if "value" in es and not isinstance(es["value"], (int, float)):
                            errors.append(
                                f"comparisons[{i}].effect_size.value must be numeric"
                            )
                    elif not isinstance(es, (int, float)):
                        errors.append(
                            f"comparisons[{i}].effect_size must be numeric or dict"
                        )

    if "test_results" in spec:
        test_results = spec["test_results"]
        if not isinstance(test_results, (dict, list)):
            errors.append("'test_results' must be a dictionary or list")

    if "descriptive" in spec:
        descriptive = spec["descriptive"]
        if not isinstance(descriptive, dict):
            errors.append("'descriptive' must be a dictionary")

    return errors


def _is_stats_zip(path: Path) -> bool:
    """True iff ``path`` ends with ``.stats.zip``."""
    suffixes = [s.lower() for s in path.suffixes]
    return suffixes[-2:] == [".stats", ".zip"]


def _write_dir_bundle(data: Dict[str, Any], dir_path: Path) -> None:
    """Write the canonical bundle layout under ``dir_path`` (must already exist)."""
    spec = data.get("spec", {})
    with open(dir_path / "stats.json", "w") as f:
        json.dump(spec, f, indent=2)

    if "data" in data:
        data_file = dir_path / "data.csv"
        df = data["data"]
        if hasattr(df, "to_csv"):
            df.to_csv(data_file, index=False)
        else:
            with open(data_file, "w") as f:
                f.write(str(df))

    if "report" in data:
        with open(dir_path / "report.md", "w") as f:
            f.write(data["report"])


def _read_dir_bundle(dir_path: Path) -> Dict[str, Any]:
    """Read the canonical bundle layout from ``dir_path``."""
    result: Dict[str, Any] = {}

    spec_file = dir_path / "stats.json"
    if spec_file.exists():
        with open(spec_file) as f:
            result["spec"] = json.load(f)
    else:
        result["spec"] = None

    data_file = dir_path / "data.csv"
    if data_file.exists():
        try:
            import pandas as pd

            result["data"] = pd.read_csv(data_file)
        except ImportError:
            with open(data_file) as f:
                result["data"] = f.read()

    report_file = dir_path / "report.md"
    if report_file.exists():
        result["report"] = report_file.read_text()

    return result


def save_stats_bundle(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save a ``.stats`` bundle to ``path``.

    ``path`` may be either a directory (legacy form — also accepts paths
    ending in ``.stats``) or a single-file ``.stats.zip`` archive. The
    return value is the actual on-disk path written.
    """
    p = Path(path)

    if _is_stats_zip(p):
        # Zipped bundle: write to a tempdir, then pack.
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "bundle"
            staging.mkdir()
            _write_dir_bundle(data, staging)

            p.parent.mkdir(parents=True, exist_ok=True)
            root = p.name[: -len(".stats.zip")]
            with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in staging.rglob("*"):
                    if f.is_file():
                        zf.write(f, Path(root) / f.relative_to(staging))
        return p

    # Directory bundle.
    p.mkdir(parents=True, exist_ok=True)
    _write_dir_bundle(data, p)
    return p


def load_stats_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a ``.stats`` bundle from ``path`` (directory or ``.stats.zip``)."""
    p = Path(path)

    if _is_stats_zip(p):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "extract"
            staging.mkdir()
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(staging)
            # Bundle root is the first child directory (matches save layout).
            children = [c for c in staging.iterdir() if c.is_dir()]
            root = children[0] if children else staging
            return _read_dir_bundle(root)

    return _read_dir_bundle(p)


# EOF
