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


def test_mcp_install_emits_config_snippet_exit_code():
    # Arrange
    # Act
    result = _invoke("mcp", "install")
    # Assert
    assert result.exit_code == 0

def test_mcp_install_emits_config_snippet_scitex_stats_stdout():
    # Arrange
    # Act
    result = _invoke("mcp", "install")
    # Assert
    assert "scitex-stats" in result.stdout


def test_mcp_install_json_exit_code():
    # Arrange
    # Act
    result = _invoke("mcp", "install", "--json")
    # Assert
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

def test_mcp_install_json_scitex_stats_payload_package():
    # Arrange
    result = _invoke("mcp", "install", "--json")
    # Act
    payload = json.loads(result.stdout)
    # Assert
    assert payload["package"] == "scitex-stats"


def test_mcp_show_installation_deprecated_alias_returns_two_exit_code():
    """Old verb should exit 2 with a redirect message."""
    # Arrange
    # Act
    result = _invoke("mcp", "show-installation")
    # Assert
    assert result.exit_code == 2

def test_mcp_show_installation_deprecated_alias_returns_two_renamed_install_lower_stderr():
    """Old verb should exit 2 with a redirect message."""
    # Arrange
    # Act
    result = _invoke("mcp", "show-installation")
    # Assert
    assert "renamed" in result.stderr.lower() or "install" in result.stderr.lower()


def test_mcp_start_dry_run_does_not_launch_server_exit_code():
    # Arrange
    # Act
    result = _invoke("mcp", "start", "--dry-run")
    # Assert
    assert result.exit_code == 0

def test_mcp_start_dry_run_does_not_launch_server_stdout():
    # Arrange
    # Act
    result = _invoke("mcp", "start", "--dry-run")
    # Assert
    assert "DRY RUN" in result.stdout


def test_mcp_list_tools_runs_via_cli_exit_code():
    # Arrange
    # Act
    result = _invoke("mcp", "list-tools")
    # Assert
    assert result.exit_code == 0

def test_mcp_list_tools_runs_via_cli_run_test_stdout():
    # Arrange
    # Act
    result = _invoke("mcp", "list-tools")
    # Assert
    assert "run_test" in result.stdout


# ----- tests group ----------------------------------------------------- #


def test_tests_list_json_exit_code():
    # Arrange
    # Act
    result = _invoke("tests", "list")
    # Assert
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)

def test_tests_list_json_parsed():
    # Arrange
    result = _invoke("tests", "list")
    # Act
    parsed = json.loads(result.stdout)
    # Assert
    assert isinstance(parsed, list) and parsed


def test_tests_list_no_json_exit_code():
    # Arrange
    # Act
    result = _invoke("tests", "list", "--no-json")
    # Assert
    assert result.exit_code == 0

def test_tests_list_no_json_splitlines_strip_stdout():
    # Arrange
    # Act
    result = _invoke("tests", "list", "--no-json")
    # Assert
    assert result.stdout.strip().splitlines()


def test_tests_describe_csv_exit_code(tmp_path):
    # Arrange
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(p, index=False)
    # Act
    result = _invoke("tests", "describe", str(p), "-c", "a")
    # Assert
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

def test_tests_describe_csv_mean_payload(tmp_path):
    # Arrange
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(p, index=False)
    result = _invoke("tests", "describe", str(p), "-c", "a")
    # Act
    payload = json.loads(result.stdout)
    # Assert
    assert "mean" in payload


def test_tests_execute_ttest_ind_exit_code(tmp_path):
    # Arrange
    p = tmp_path / "two.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0.5, 1, 30)}).to_csv(
        p, index=False
    )
    # Act
    result = _invoke("tests", "execute", "ttest_ind", str(p), "--x", "a", "--y", "b")
    # Assert
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

def test_tests_execute_ttest_ind_value_payload_pvalue(tmp_path):
    # Arrange
    p = tmp_path / "two.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0.5, 1, 30)}).to_csv(
        p, index=False
    )
    result = _invoke("tests", "execute", "ttest_ind", str(p), "--x", "a", "--y", "b")
    # Act
    payload = json.loads(result.stdout)
    # Assert
    assert "p_value" in payload or "pvalue" in payload


def test_tests_recommend_runs_via_cli_exit_code():
    # Arrange
    # Act
    result = _invoke(
        "tests",
        "recommend",
        "--n-groups",
        "2",
        "--sample-sizes",
        "30,30",
    )
    # Assert
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)

def test_tests_recommend_runs_via_cli_parsed_list():
    # Arrange
    result = _invoke(
        "tests",
        "recommend",
        "--n-groups",
        "2",
        "--sample-sizes",
        "30,30",
    )
    # Act
    parsed = json.loads(result.stdout)
    # Assert
    assert isinstance(parsed, list)


# ----- format-pvalue --------------------------------------------------- #


def test_format_pvalue_significant_exit_code():
    # Arrange
    # Act
    result = _invoke("format-pvalue", "0.001")
    # Assert
    assert result.exit_code == 0

def test_format_pvalue_significant_strip_stdout():
    # Arrange
    # Act
    result = _invoke("format-pvalue", "0.001")
    # Assert
    assert result.stdout.strip()


def test_format_pvalue_with_style():
    # Arrange
    # Act
    result = _invoke("format-pvalue", "0.04", "--style", "apa")
    # Assert
    assert result.exit_code == 0


# ----- top-level --------------------------------------------------------- #


def test_help_recursive_lists_subcommands_exit_code():
    # Arrange
    # Act
    result = _invoke("--help-recursive")
    # Assert
    assert result.exit_code == 0
    text = result.stdout

def test_help_recursive_lists_subcommands_mcp_text_tests():
    # Arrange
    # Act
    result = _invoke("--help-recursive")
    text = result.stdout
    # Assert
    assert "mcp" in text and "tests" in text


def test_version_flag_exit_code():
    # Arrange
    # Act
    result = _invoke("--version")
    # Assert
    assert result.exit_code == 0

def test_version_flag_strip_stdout():
    # Arrange
    # Act
    result = _invoke("--version")
    # Assert
    assert result.stdout.strip()


def test_main_no_args_prints_help_exit_code():
    # Arrange
    # Act
    result = _invoke()
    # Assert
    assert result.exit_code in (0, 2)
    combined = result.stdout + result.stderr

def test_main_no_args_prints_help_mcp_combined_tests():
    # Arrange
    # Act
    result = _invoke()
    combined = result.stdout + result.stderr
    # Assert
    assert "mcp" in combined or "tests" in combined
