"""Smoke test for examples/03_multiple_comparison.ipynb.

PS-505: notebook smoke-tests must drive `jupyter nbconvert --execute`
(or pytest-nbval) so cells actually run end-to-end.
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "examples" / "03_multiple_comparison.ipynb"
)


def test_notebook_exists():
    assert NOTEBOOK.exists(), f"missing notebook: {NOTEBOOK}"


def test_notebook_executes(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            str(tmp_path / "out.ipynb"),
            str(NOTEBOOK),
        ],
        check=True,
        timeout=180,
    )
