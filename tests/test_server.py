#!/usr/bin/env python3
# File: tests/test_server.py

"""Tests for the MCP server creation and tool listing.

Verifies that the FastMCP server is properly configured with
all expected tools and that tools have correct signatures.
"""

import asyncio

import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture(scope="module")
def mcp_server():
    """Return the FastMCP server instance."""
    from scitex_stats._server import mcp

    return mcp


@pytest.fixture(scope="module")
def tool_names(mcp_server):
    """Return sorted list of registered tool names."""

    async def _list():
        tools = await mcp_server.list_tools()
        return sorted(t.name for t in tools)

    return _run_async(_list())


@pytest.fixture(scope="module")
def tools_by_name(mcp_server):
    """Return dict of tool_name -> tool object."""

    async def _list():
        tools = await mcp_server.list_tools()
        return {t.name: t for t in tools}

    return _run_async(_list())


# ===================================================================
# 1. Server creation
# ===================================================================


class TestServerCreation:
    """Test that the MCP server is properly created."""

    def test_server_exists(self, mcp_server):
        assert mcp_server is not None

    def test_server_name(self, mcp_server):
        assert mcp_server.name == "scitex-stats"

    def test_server_has_instructions(self, mcp_server):
        assert mcp_server.instructions is not None
        assert len(mcp_server.instructions) > 0


# ===================================================================
# 2. Tool listing
# ===================================================================

EXPECTED_TOOLS = [
    "correct_pvalues",
    "describe",
    "effect_size",
    "format_results",
    "normality_test",
    "p_to_stars",
    "posthoc_test",
    "power_analysis",
    "recommend_tests",
    "run_test",
]


class TestToolListing:
    """Test that all expected tools are registered."""

    def test_tool_count(self, tool_names):
        assert len(tool_names) == 10, (
            f"Expected 10 tools, got {len(tool_names)}: {tool_names}"
        )

    @pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
    def test_tool_registered(self, tool_name, tool_names):
        assert tool_name in tool_names, f"Tool '{tool_name}' not found in {tool_names}"

    def test_all_tools_have_descriptions(self, tools_by_name):
        for name, tool in tools_by_name.items():
            assert tool.description is not None and len(tool.description) > 0, (
                f"Tool '{name}' has no description"
            )


# ===================================================================
# 3. Tool schemas
# ===================================================================


class TestToolSchemas:
    """Test that tools have proper input schemas (via FastMCP parameters attr)."""

    def test_run_test_has_test_name_param(self, tools_by_name):
        tool = tools_by_name["run_test"]
        props = tool.parameters.get("properties", {})
        assert "test_name" in props

    def test_recommend_tests_has_n_groups(self, tools_by_name):
        tool = tools_by_name["recommend_tests"]
        props = tool.parameters.get("properties", {})
        assert "n_groups" in props

    def test_correct_pvalues_has_pvalues(self, tools_by_name):
        tool = tools_by_name["correct_pvalues"]
        props = tool.parameters.get("properties", {})
        assert "pvalues" in props

    def test_effect_size_has_groups(self, tools_by_name):
        tool = tools_by_name["effect_size"]
        props = tool.parameters.get("properties", {})
        assert "group1" in props
        assert "group2" in props

    def test_p_to_stars_has_p_value(self, tools_by_name):
        tool = tools_by_name["p_to_stars"]
        props = tool.parameters.get("properties", {})
        assert "p_value" in props


# ===================================================================
# 4. run_server function
# ===================================================================


class TestRunServerFunction:
    """Test that run_server is importable and callable."""

    def test_run_server_importable(self):
        from scitex_stats._server import run_server

        assert callable(run_server)

    def test_mcp_module_re_exports(self):
        from scitex_stats._mcp import mcp, run_server

        assert mcp is not None
        assert callable(run_server)


# EOF
