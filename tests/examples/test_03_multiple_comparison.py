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


def test_notebook_exists_case():
    # Arrange
    # Act
    # Assert
    assert NOTEBOOK.exists(), f"missing notebook: {NOTEBOOK}"


def test_notebook_executes_without_error(tmp_path):
    # Arrange
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        # nbclient's kernel-handshake default (60s) is too tight for a
        # heavy `import scitex_stats` (numpy/scipy/pandas/matplotlib
        # font-cache build) under CI load — was intermittently raising
        # "Kernel didn't respond in 60 seconds" unrelated to notebook
        # content. Stay comfortably under the outer subprocess timeout.
        "--ExecutePreprocessor.startup_timeout=120",
        "--output",
        str(tmp_path / "out.ipynb"),
        str(NOTEBOOK),
    ]
    # Act
    result = subprocess.run(cmd, check=True, timeout=180)
    # Assert
    assert result.returncode == 0
