"""Smoke tests: every example notebook must execute end-to-end.

PS-505: drive `jupyter nbconvert --execute` so cells actually run.
"""

import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS = sorted(
    Path(__file__).resolve().parents[2].joinpath("examples").glob("*.ipynb")
)


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes_without_error(nb, tmp_path):
    # Arrange
    cmd = [
        sys.executable,
        "-m",
        # `-m nbconvert` (not `-m jupyter nbconvert`): the latter goes
        # through jupyter_core's subcommand dispatch, which resolves
        # `jupyter-nbconvert` by searching PATH — NOT via sys.path — so
        # on a machine with a stray global `~/.local/bin/jupyter-nbconvert`
        # ahead of this venv on PATH, it launches that Python instead of
        # this venv's, crashing with ModuleNotFoundError: nbconvert.
        # `-m nbconvert` uses Python's own sys.path-based module
        # resolution, always the interpreter actually running this test.
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
        str(tmp_path / nb.name),
        str(nb),
    ]
    # Act
    result = subprocess.run(cmd, check=True, timeout=180)
    # Assert
    assert result.returncode == 0
