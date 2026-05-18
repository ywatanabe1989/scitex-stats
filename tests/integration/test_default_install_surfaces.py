"""Smoke tests: each of the 4 public surfaces must work in a default install.

Default install = `pip install scitex-stats` (no extras). Per the
dep-cleanup audit (feat/pyproject-deps-cleanup), all 4 surfaces
— Python API, CLI, MCP server, Skills entry-points — are required to
import / start without any `[extra]` installed.

These tests guard against regressions where a future change pushes a
required dep back into an extra and silently breaks a surface for
fresh installs. They do **not** exercise feature depth — that's the
job of the per-test suites.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest


# ----- Surface 1: Python API -------------------------------------------------


def test_python_api_top_level_import():
    """`import scitex_stats` must succeed and expose __version__."""
    import scitex_stats

    assert isinstance(scitex_stats.__version__, str)
    assert scitex_stats.__version__  # non-empty


def test_python_api_describe_runs():
    """The headline `describe()` entry should run on a trivial input."""
    from scitex_stats import describe

    out = describe([1.0, 2.0, 3.0])
    assert out is not None


def test_python_api_run_test_ttest_ind():
    """`run_test('ttest_ind', a, b)` should return a dict with a p-value."""
    import numpy as np

    import scitex_stats as sst

    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 30)
    b = rng.normal(0.5, 1.0, 30)
    result = sst.run_test("ttest_ind", a, b)
    assert isinstance(result, dict)
    assert "pvalue" in result or "p_value" in result


# ----- Surface 2: CLI --------------------------------------------------------


def test_cli_help_runs():
    """`scitex-stats --help` exits 0."""
    exe = shutil.which("scitex-stats")
    assert exe, "scitex-stats CLI not on PATH"
    proc = subprocess.run([exe, "--help"], capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"scitex-stats" in proc.stdout


def test_cli_version_runs():
    """`scitex-stats --version` prints a version line."""
    exe = shutil.which("scitex-stats")
    assert exe
    proc = subprocess.run([exe, "--version"], capture_output=True, timeout=30)
    assert proc.returncode == 0
    assert b"version" in proc.stdout.lower()


def test_cli_tests_list_runs():
    """`scitex-stats tests list` returns the 23-test catalog."""
    exe = shutil.which("scitex-stats")
    assert exe
    proc = subprocess.run(
        [exe, "tests", "list"], capture_output=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    # Catalog includes core names
    out = proc.stdout.decode()
    for name in ("ttest_ind", "anova", "pearson", "chi2", "shapiro"):
        assert name in out, f"missing {name} in tests list output"


# ----- Surface 3: MCP server -------------------------------------------------


def test_mcp_server_module_imports():
    """`from scitex_stats._server import mcp, run_server` must succeed.

    This is the canonical "MCP works out-of-the-box" check — proves the
    fastmcp + mcp deps are in the default install, not gated behind an
    extra.
    """
    from scitex_stats._server import mcp, run_server  # noqa: F401

    assert mcp is not None
    assert callable(run_server)


def test_mcp_doctor_clean():
    """`scitex-stats mcp doctor` reports all checks passing."""
    exe = shutil.which("scitex-stats")
    assert exe
    proc = subprocess.run(
        [exe, "mcp", "doctor"], capture_output=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"All checks passed" in proc.stdout


# ----- Surface 4: Skills + scitex-dev entry-points ---------------------------


@pytest.mark.parametrize(
    "group, expected_name",
    [
        ("scitex_dev.docs", "scitex-stats"),
        ("scitex_dev.skills", "scitex-stats"),
        ("scitex_dev.linter.plugins", "stats"),
    ],
)
def test_scitex_dev_entry_points_registered(group, expected_name):
    """scitex-stats publishes 3 scitex-dev entry-points; all must resolve."""
    from importlib.metadata import entry_points

    eps = entry_points(group=group)
    names = {e.name for e in eps}
    assert expected_name in names, (
        f"entry-point group {group!r} missing {expected_name!r}; "
        f"have: {sorted(names)}"
    )


def test_linter_plugin_loads():
    """The linter plugin must be importable + return a non-empty rule list."""
    from scitex_stats._linter_plugin import get_plugin

    plugin = get_plugin()
    assert isinstance(plugin, dict)
    assert plugin.get("rules"), "linter plugin returned no rules"


def test_skills_cli_list_runs():
    """`scitex-stats skills list` enumerates bundled skill files."""
    exe = shutil.which("scitex-stats")
    assert exe
    proc = subprocess.run(
        [exe, "skills", "list"], capture_output=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    out = proc.stdout.decode()
    # SKILL.md is the mandatory front-door skill
    assert "SKILL" in out


# ----- Bonus: import-time hygiene -------------------------------------------


def test_top_level_import_does_not_load_heavy_stack():
    """Cold `import scitex_stats` must NOT eagerly load numpy/scipy/etc.

    PEP 562 lazy-loading in scitex_stats/__init__.py keeps CLI startup
    fast (see audit-cli rule §10). A regression here means a heavy
    submodule was re-exported eagerly.
    """
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import scitex_stats\n"
        "after = set(sys.modules) - before\n"
        "heavy = sorted(\n"
        "    m for m in after\n"
        "    if m.split('.')[0] in {\n"
        "        'numpy', 'scipy', 'pandas', 'statsmodels',\n"
        "        'matplotlib', 'figrecipe', 'pingouin', 'torch',\n"
        "    }\n"
        ")\n"
        "print(','.join(heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    loaded = proc.stdout.decode().strip()
    assert loaded == "", (
        f"cold `import scitex_stats` eagerly loaded heavy deps: {loaded!r}"
    )
