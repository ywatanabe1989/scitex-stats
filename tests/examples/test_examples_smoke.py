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
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--output",
        str(tmp_path / nb.name),
        str(nb),
    ]
    # Act
    result = subprocess.run(cmd, check=True, timeout=180)
    # Assert
    assert result.returncode == 0
