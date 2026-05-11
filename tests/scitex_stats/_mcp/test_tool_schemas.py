"""Direct test for `scitex_stats._mcp.tool_schemas.get_tool_schemas`.

The module is mostly a single big `return [...]` of `mcp.types.Tool`
instances. Coverage was 0 % only because nothing imported and called
it. Calling it once verifies the shape and registers the lines.
"""

from __future__ import annotations

import pytest

mcp_types = pytest.importorskip("mcp.types")

from scitex_stats._mcp.tool_schemas import get_tool_schemas

EXPECTED_TOOLS = {
    "recommend_tests",
    "run_test",
    "format_results",
    "power_analysis",
    "correct_pvalues",
    "describe",
    "effect_size",
    "normality_test",
    "posthoc_test",
    "p_to_stars",
}


def test_get_tool_schemas_returns_list_of_tools():
    tools = get_tool_schemas()
    assert isinstance(tools, list) and tools
    assert all(isinstance(t, mcp_types.Tool) for t in tools)


def test_get_tool_schemas_covers_every_expected_tool():
    tools = get_tool_schemas()
    names = {t.name for t in tools}
    missing = EXPECTED_TOOLS - names
    assert not missing, f"missing tool schemas: {missing}"


def test_each_schema_has_description_and_input_schema():
    for t in get_tool_schemas():
        assert t.description and isinstance(t.description, str), (
            f"{t.name} has no description"
        )
        assert t.inputSchema and isinstance(t.inputSchema, dict), (
            f"{t.name} has no inputSchema"
        )
        # Every Tool's inputSchema is an object schema with properties
        assert t.inputSchema.get("type") == "object", (
            f"{t.name} inputSchema is not an object"
        )
        assert "properties" in t.inputSchema, f"{t.name} inputSchema lacks properties"


def test_run_test_schema_lists_all_23_tests():
    """The `run_test` schema's `test_name` enum is the source of truth
    for which tests the MCP surface accepts. Catch additions / removals
    early."""
    schema = next(t for t in get_tool_schemas() if t.name == "run_test").inputSchema
    enum_vals = schema["properties"]["test_name"]["enum"]
    # Currently 23 tests across parametric / nonparametric / correlation /
    # categorical / normality categories.
    assert len(enum_vals) == 23, f"expected 23 tests, got {len(enum_vals)}"
