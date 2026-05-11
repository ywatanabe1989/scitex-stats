"""Drive every click subcommand in `_cli/__init__.py` via CliRunner.

The worker functions (`_run_tests_list`, `_cmd_start`, etc.) are
already covered by their own test files; this file covers the click
command wrappers that glue them together.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from scitex_stats._cli import main


def _invoke(*args):
    return CliRunner().invoke(main, list(args))


# ----- mcp ------------------------------------------------------------- #


def test_mcp_install_emits_config_snippet():
    result = _invoke("mcp", "install")
    assert result.exit_code == 0
    assert "scitex-stats" in result.stdout


def test_mcp_install_json():
    result = _invoke("mcp", "install", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["package"] == "scitex-stats"


def test_mcp_show_installation_deprecated_alias_returns_two():
    """Old verb should exit 2 with a redirect message."""
    result = _invoke("mcp", "show-installation")
    assert result.exit_code == 2
    assert "renamed" in result.stderr.lower() or "install" in result.stderr.lower()


def test_mcp_start_dry_run_does_not_launch_server():
    result = _invoke("mcp", "start", "--dry-run")
    assert result.exit_code == 0
    assert "DRY RUN" in result.stdout


def test_mcp_list_tools_runs_via_cli():
    result = _invoke("mcp", "list-tools")
    assert result.exit_code == 0
    assert "run_test" in result.stdout


# ----- tests group ----------------------------------------------------- #


def test_tests_list_json():
    result = _invoke("tests", "list")
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list) and parsed


def test_tests_list_no_json():
    result = _invoke("tests", "list", "--no-json")
    assert result.exit_code == 0
    # Plain-text output: one test per line, at least one.
    assert result.stdout.strip().splitlines()


def test_tests_describe_csv(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(p, index=False)
    result = _invoke("tests", "describe", str(p), "-c", "a")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "mean" in payload


def test_tests_execute_ttest_ind(tmp_path):
    p = tmp_path / "two.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0.5, 1, 30)}).to_csv(
        p, index=False
    )
    result = _invoke("tests", "execute", "ttest_ind", str(p), "--x", "a", "--y", "b")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "p_value" in payload or "pvalue" in payload


def test_tests_recommend_runs_via_cli():
    result = _invoke(
        "tests",
        "recommend",
        "--n-groups",
        "2",
        "--sample-sizes",
        "30,30",
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)


# ----- format-pvalue --------------------------------------------------- #


def test_format_pvalue_significant():
    result = _invoke("format-pvalue", "0.001")
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_format_pvalue_with_style():
    result = _invoke("format-pvalue", "0.04", "--style", "apa")
    assert result.exit_code == 0


# ----- top-level --------------------------------------------------------- #


def test_help_recursive_lists_subcommands():
    result = _invoke("--help-recursive")
    assert result.exit_code == 0
    text = result.stdout
    assert "mcp" in text and "tests" in text


def test_version_flag():
    result = _invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_main_no_args_prints_help():
    result = _invoke()
    # With no args click groups exit 0 if `invoke_without_command=True`
    # or 2 otherwise — either is acceptable as long as help body landed.
    assert result.exit_code in (0, 2)
    combined = result.stdout + result.stderr
    assert "mcp" in combined or "tests" in combined
