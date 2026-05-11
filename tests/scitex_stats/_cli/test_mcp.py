"""Direct tests for `scitex_stats._cli.mcp`.

The integration test_cli.py exercises `mcp list-tools` and `mcp
doctor` via subprocess, but several branches xfail because the CLI
uses internals that broke in FastMCP 3.x (`_tool_manager`). These
tests cover the parts that don't depend on that API:

- `_get_tool_module` (pure dispatch by name)
- `_style` (no-op when stdout isn't a tty)
- `_format_tool_signature` (with a duck-typed fake tool)
- `cmd_start(dry_run=True)` (no real server start)
- `cmd_config` (text + JSON snippets)
- `cmd_doctor` (returns 0 or 1; doesn't crash)
"""

from __future__ import annotations

import importlib
import json

import pytest

# `from scitex_stats._cli import mcp` resolves to the Click `mcp`
# Group that `_cli/__init__.py` rebinds at line ~138, not the
# `_cli/mcp.py` module file. Reach the file under its package path.
cli_mcp = importlib.import_module("scitex_stats._cli.mcp")

# ----- _get_tool_module ------------------------------------------------- #


@pytest.mark.parametrize(
    "tool_name,expected_module",
    [
        ("recommend_tests", "auto"),
        ("correct_pvalues", "correct"),
        ("posthoc_test", "posthoc"),
        ("power_analysis", "power"),
        ("effect_size", "effect_sizes"),
        ("normality_test", "normality"),
        ("describe", "descriptive"),
        ("format_results", "formatting"),
        ("p_to_stars", "formatting"),
        ("run_test", "general"),
    ],
)
def test_get_tool_module_dispatch(tool_name, expected_module):
    assert cli_mcp._get_tool_module(tool_name) == expected_module


# ----- _style ----------------------------------------------------------- #


def test_style_passthrough_when_not_tty(capsys):
    # Running under pytest, sys.stdout isn't a tty → _style returns text as-is
    assert cli_mcp._style("hello") == "hello"
    assert cli_mcp._style("hello", fg="green", bold=True) == "hello"


# ----- _format_tool_signature ------------------------------------------ #


class _FakeTool:
    """Duck-typed stand-in for a fastmcp Tool."""

    def __init__(self, name, parameters, fn=None, description=None):
        self.name = name
        self.parameters = parameters
        self.fn = fn
        self.description = description


def test_format_tool_signature_compact_form():
    tool = _FakeTool(
        name="my_tool",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        },
    )
    out = cli_mcp._format_tool_signature(tool, compact=True)
    assert "my_tool" in out
    assert "x" in out


def test_format_tool_signature_multiline_for_many_params():
    tool = _FakeTool(
        name="big_tool",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "string", "default": "hi"},
                "c": {"type": "integer"},
                "d": {"type": "boolean"},
            },
            "required": ["a"],
        },
    )
    out = cli_mcp._format_tool_signature(tool, compact=False)
    assert "big_tool" in out
    # Multiline mode introduces newlines between params
    assert "\n" in out


# ----- cmd_start (dry-run) --------------------------------------------- #


def test_cmd_start_dry_run_does_not_run_server(capsys):
    rc = cli_mcp.cmd_start(dry_run=True)
    captured = capsys.readouterr()
    assert rc == 0
    assert "DRY RUN" in captured.out
    assert "stdio" in captured.out


def test_cmd_start_dry_run_honours_transport(capsys):
    rc = cli_mcp.cmd_start(dry_run=True, transport="sse")
    assert rc == 0
    assert "sse" in capsys.readouterr().out


# ----- cmd_config ------------------------------------------------------- #


def test_cmd_config_text_form_prints_snippets(capsys):
    rc = cli_mcp.cmd_config(as_json=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "scitex-stats" in out
    # Either or both config snippets should be referenced
    assert "Option 1" in out or "claude_desktop_config" in out


def test_cmd_config_json_form_is_parseable(capsys):
    rc = cli_mcp.cmd_config(as_json=True)
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["package"] == "scitex-stats"
    assert "snippets" in payload
    assert "cli" in payload["snippets"]
    assert "python_module" in payload["snippets"]
    assert "config_paths" in payload


# ----- cmd_doctor ------------------------------------------------------- #


def test_cmd_doctor_returns_zero_or_one(capsys):
    rc = cli_mcp.cmd_doctor()
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert "Health Check" in out
    # Always reports on fastmcp + MCP server + CLI
    assert "fastmcp" in out
    assert "CLI" in out


# ----- cmd_list_tools (in-process, FastMCP 3.x compatible) ----------- #


def test_cmd_list_tools_text_output(capsys):
    rc = cli_mcp.cmd_list_tools(verbose=0)
    out = capsys.readouterr().out
    assert rc == 0
    assert "SciTeX Stats MCP" in out
    # verbose=0 emits one tool per line
    assert "recommend_tests" in out
    assert "run_test" in out


def test_cmd_list_tools_json_output(capsys):
    import json as _json

    rc = cli_mcp.cmd_list_tools(verbose=0, as_json=True)
    out = capsys.readouterr().out
    assert rc == 0
    payload = _json.loads(out)
    assert payload["name"] == "scitex-stats"
    assert payload["total"] >= 10
    assert "modules" in payload


def test_cmd_list_tools_verbose_1_emits_signatures(capsys):
    rc = cli_mcp.cmd_list_tools(verbose=1)
    out = capsys.readouterr().out
    assert rc == 0
    # verbose=1 prints function-like signatures; expect param names
    assert "test_name" in out or "data" in out


def test_cmd_list_tools_verbose_2_emits_descriptions(capsys):
    rc = cli_mcp.cmd_list_tools(verbose=2, compact=True)
    out = capsys.readouterr().out
    assert rc == 0
    # verbose=2 also prints the first description line per tool
    assert "p_value" in out or "Statistical" in out or "test" in out


def test_cmd_list_tools_verbose_3_emits_full_descriptions(capsys):
    rc = cli_mcp.cmd_list_tools(verbose=3, compact=True)
    out = capsys.readouterr().out
    assert rc == 0
    # verbose=3 prints multi-line descriptions
    assert out


def test_cmd_list_tools_module_filter_accepts_known_module(capsys):
    rc = cli_mcp.cmd_list_tools(verbose=0, module_filter="auto")
    out = capsys.readouterr().out
    assert rc == 0
    assert "recommend_tests" in out


def test_cmd_list_tools_module_filter_rejects_unknown(capsys):
    rc = cli_mcp.cmd_list_tools(verbose=0, module_filter="not_a_module")
    out = capsys.readouterr().out
    assert rc == 1
    assert "Unknown module" in out
    assert "Available modules" in out
