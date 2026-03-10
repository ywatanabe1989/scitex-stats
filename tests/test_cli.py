#!/usr/bin/env python3
# File: tests/test_cli.py

"""Tests for scitex-stats CLI commands.

Verifies that all CLI subcommands work via subprocess and via
the internal main() entry point.
"""

import subprocess
import sys

import pytest

PYTHON = sys.executable
CLI_MODULE = [PYTHON, "-m", "scitex_stats"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_cli(*args, timeout=30):
    """Run scitex-stats CLI and return CompletedProcess."""
    cmd = CLI_MODULE + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ===================================================================
# 1. Top-level commands
# ===================================================================


class TestTopLevelCLI:
    """Test top-level CLI flags."""

    def test_help(self):
        result = _run_cli("--help")
        assert result.returncode == 0
        assert (
            "scitex-stats" in result.stdout.lower() or "scitex" in result.stdout.lower()
        )
        assert "mcp" in result.stdout

    def test_version(self):
        result = _run_cli("-V")
        assert result.returncode == 0
        assert "scitex-stats" in result.stdout

    def test_help_recursive(self):
        result = _run_cli("--help-recursive")
        assert result.returncode == 0
        assert "mcp" in result.stdout
        assert "list-tools" in result.stdout or "list_tools" in result.stdout

    def test_no_args_shows_help(self):
        result = _run_cli()
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "scitex" in result.stdout.lower()


# ===================================================================
# 2. list-python-apis
# ===================================================================


class TestListPythonAPIs:
    """Test list-python-apis subcommand."""

    def test_list_python_apis(self):
        result = _run_cli("list-python-apis")
        assert result.returncode == 0
        assert "scitex_stats" in result.stdout or "API" in result.stdout

    def test_list_python_apis_json(self):
        result = _run_cli("list-python-apis", "--json")
        assert result.returncode == 0
        # Should be valid JSON
        import json

        data = json.loads(result.stdout)
        assert isinstance(data, list)


# ===================================================================
# 3. mcp subcommands
# ===================================================================


class TestMCPCLI:
    """Test MCP-related CLI subcommands."""

    def test_mcp_help(self):
        result = _run_cli("mcp")
        assert result.returncode == 0

    @pytest.mark.xfail(
        reason="CLI uses _tool_manager which does not exist in FastMCP 3.x",
        strict=False,
    )
    def test_mcp_list_tools(self):
        result = _run_cli("mcp", "list-tools")
        assert result.returncode == 0
        assert "recommend_tests" in result.stdout or "run_test" in result.stdout

    @pytest.mark.xfail(
        reason="CLI uses _tool_manager which does not exist in FastMCP 3.x",
        strict=False,
    )
    def test_mcp_list_tools_json(self):
        result = _run_cli("mcp", "list-tools", "--json")
        assert result.returncode == 0
        import json

        data = json.loads(result.stdout)
        assert "tools" in str(data)

    @pytest.mark.xfail(
        reason="CLI uses _tool_manager which does not exist in FastMCP 3.x",
        strict=False,
    )
    def test_mcp_doctor(self):
        result = _run_cli("mcp", "doctor", timeout=15)
        # doctor may return 0 or 1 depending on environment
        assert result.returncode in (0, 1)
        assert "Health Check" in result.stdout or "fastmcp" in result.stdout


# ===================================================================
# 4. Internal main() entry point
# ===================================================================


class TestMainEntryPoint:
    """Test the CLI main() function directly."""

    def test_main_help_returns_zero(self):
        from scitex_stats._cli import main

        ret = main(["--help-recursive"])
        assert ret == 0

    def test_main_no_args_returns_zero(self):
        from scitex_stats._cli import main

        ret = main([])
        assert ret == 0

    def test_main_version(self):
        from scitex_stats._cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["-V"])
        assert exc_info.value.code == 0


# EOF
