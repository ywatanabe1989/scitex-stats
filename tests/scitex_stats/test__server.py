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


async def _client_list_tools(mcp_server):
    """Enumerate tools via the in-memory FastMCP client.

    FastMCP 2.12 removed the public ``FastMCP.list_tools()`` coroutine;
    the supported enumeration path is the in-memory ``Client`` transport.
    """
    from fastmcp import Client

    async with Client(mcp_server) as client:
        return await client.list_tools()


@pytest.fixture(scope="module")
def tool_names(mcp_server):
    """Return sorted list of registered tool names."""
    tools = _run_async(_client_list_tools(mcp_server))
    return sorted(t.name for t in tools)


@pytest.fixture(scope="module")
def tools_by_name(mcp_server):
    """Return dict of tool_name -> tool object."""
    tools = _run_async(_client_list_tools(mcp_server))
    return {t.name: t for t in tools}


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
        assert len(tool_names) >= 10, (
            f"Expected at least 10 tools, got {len(tool_names)}: {tool_names}"
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
        props = tool.inputSchema.get("properties", {})
        assert "test_name" in props

    def test_recommend_tests_has_n_groups(self, tools_by_name):
        tool = tools_by_name["recommend_tests"]
        props = tool.inputSchema.get("properties", {})
        assert "n_groups" in props

    def test_correct_pvalues_has_pvalues(self, tools_by_name):
        tool = tools_by_name["correct_pvalues"]
        props = tool.inputSchema.get("properties", {})
        assert "pvalues" in props

    def test_effect_size_has_groups(self, tools_by_name):
        tool = tools_by_name["effect_size"]
        props = tool.inputSchema.get("properties", {})
        assert "group1" in props
        assert "group2" in props

    def test_p_to_stars_has_p_value(self, tools_by_name):
        tool = tools_by_name["p_to_stars"]
        props = tool.inputSchema.get("properties", {})
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


# ===================================================================
# 5. Direct invocation of each @mcp.tool() function
# ===================================================================
# FastMCP 2.12 wraps an @mcp.tool() target in a non-callable
# FunctionTool; the underlying coroutine is reachable via ``.fn``.
# Calling it directly bumps coverage on the tool bodies, which are
# unreachable through the registration-only tests above.

import json
import numpy as np

from scitex_stats import _server as srv


def _decode(s):
    return json.loads(s)


def test_recommend_tests_direct():
    out = _decode(
        _run_async(srv.recommend_tests.fn(n_groups=2, sample_sizes=[30, 30], top_k=3))
    )
    assert out["success"] is True


def test_run_test_direct_ttest_ind():
    rng_s = np.random.default_rng(0)
    g1 = rng_s.normal(0, 1, 30).tolist()
    g2 = rng_s.normal(0.5, 1, 30).tolist()
    out = _decode(_run_async(srv.run_test.fn(test_name="ttest_ind", data=[g1, g2])))
    assert out["success"] is True
    assert "p_value" in out


def test_format_results_direct():
    out = _decode(
        _run_async(
            srv.format_results.fn(
                test_name="ttest_ind",
                statistic=-3.21,
                p_value=0.002,
                df=58,
                effect_size=-0.83,
                effect_size_name="d",
                style="apa",
            )
        )
    )
    assert out["success"] is True
    assert "formatted" in out


def test_power_analysis_direct():
    out = _decode(
        _run_async(
            srv.power_analysis.fn(
                test_type="ttest",
                effect_size=0.5,
                power=0.8,
                alpha=0.05,
            )
        )
    )
    assert out["success"] is True


def test_correct_pvalues_direct():
    out = _decode(
        _run_async(
            srv.correct_pvalues.fn(
                pvalues=[0.001, 0.04, 0.03, 0.20, 0.005],
                method="fdr_bh",
                alpha=0.05,
            )
        )
    )
    assert isinstance(out, dict)


def test_describe_direct():
    out = _decode(_run_async(srv.describe.fn(data=[1.0, 2.0, 3.0, 4.0, 5.0])))
    assert out["success"] is True
    assert "mean" in out


def test_effect_size_direct():
    out = _decode(
        _run_async(
            srv.effect_size.fn(
                group1=[1, 2, 3, 4, 5],
                group2=[2, 3, 4, 5, 6],
                measure="cohens_d",
            )
        )
    )
    assert out["success"] is True
    assert out["measure"] == "Cohen's d"


def test_normality_test_direct():
    rng_n = np.random.default_rng(0)
    out = _decode(
        _run_async(srv.normality_test.fn(data=rng_n.normal(0, 1, 50).tolist()))
    )
    assert out["success"] is True
    assert out["test"] == "Shapiro-Wilk"


def test_posthoc_test_direct():
    rng_p = np.random.default_rng(0)
    groups = [rng_p.normal(0, 1, 25).tolist() for _ in range(3)]
    out = _decode(
        _run_async(
            srv.posthoc_test.fn(
                groups=groups,
                group_names=["A", "B", "C"],
                method="tukey",
            )
        )
    )
    assert out["success"] is True
    assert out["method"] == "tukey"


def test_p_to_stars_direct():
    out = _decode(_run_async(srv.p_to_stars.fn(p_value=0.001)))
    assert isinstance(out, dict)


def test_skills_list_direct_returns_json_envelope():
    out = _decode(_run_async(srv.skills_list.fn()))
    assert "success" in out


def test_skills_get_main_skill_direct():
    out = _decode(_run_async(srv.skills_get.fn()))
    assert "success" in out


def test_skills_get_unknown_name_direct():
    out = _decode(_run_async(srv.skills_get.fn(name="definitely-not-a-real-skill")))
    assert out["success"] is False
