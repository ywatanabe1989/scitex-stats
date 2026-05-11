"""Tests for `scitex_stats.auto._decision_tree`.

Module was at 43% — the data dict was importable but none of the
rendering helpers were called by the existing test suite.
"""

from __future__ import annotations

import shutil

import pytest

from scitex_stats.auto._decision_tree import (
    DECISION_TREE,
    DecisionNode,
    get_decision_tree,
    get_leaf_node_test_ids,
    render_flowchart_mermaid,
    render_flowchart_svg,
)


def test_decision_tree_dict_shape():
    out = get_decision_tree()
    assert isinstance(out, dict) and out
    sample = next(iter(out.values()))
    for k in ("id", "label", "shape", "emphasis", "children", "test_id"):
        assert k in sample


def test_decision_tree_module_dict_consistent_with_helper():
    helper = get_decision_tree()
    assert set(helper.keys()) == set(DECISION_TREE.keys())


def test_decision_node_dataclass_fields_round_trip():
    sample_id = next(iter(DECISION_TREE))
    node = DECISION_TREE[sample_id]
    assert isinstance(node, DecisionNode)
    # `children` is a list of (edge_label, child_id) tuples
    for edge_label, child_id in node.children:
        assert isinstance(edge_label, str)
        assert isinstance(child_id, str)


def test_get_leaf_node_test_ids_only_returns_test_carrying_nodes():
    leaves = get_leaf_node_test_ids()
    assert isinstance(leaves, dict) and leaves
    # Every leaf has a non-None test_id
    for node_id, test_id in leaves.items():
        assert DECISION_TREE[node_id].test_id == test_id
        assert test_id is not None


def test_render_flowchart_mermaid_emits_non_empty_markup():
    out = render_flowchart_mermaid()
    assert isinstance(out, str) and out.strip()
    # Mermaid flowcharts start with `flowchart` or `graph` directive
    assert "flowchart" in out.lower() or "graph" in out.lower()


def _mmdc_works() -> bool:
    """`shutil.which("mmdc")` only verifies the binary is on PATH —
    the underlying puppeteer/Chromium chain may still fail in CI
    without browsers. Probe with a minimal render."""
    if shutil.which("mmdc") is None:
        return False
    import subprocess
    import tempfile

    with (
        tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as src,
        tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as dst,
    ):
        src.write("graph TD\n  A-->B")
        src.flush()
        try:
            r = subprocess.run(
                ["mmdc", "-i", src.name, "-o", dst.name],
                capture_output=True,
                timeout=15,
            )
            return r.returncode == 0
        except Exception:
            return False


_MMDC_OK = _mmdc_works()


@pytest.mark.skipif(
    not _MMDC_OK,
    reason="render_flowchart_svg needs a working mmdc + Chromium chain",
)
def test_render_flowchart_svg_returns_svg_string():
    out = render_flowchart_svg()
    assert isinstance(out, str) and out.strip()
    assert "<svg" in out.lower()


@pytest.mark.skipif(
    not _MMDC_OK,
    reason="render_flowchart_svg needs a working mmdc + Chromium chain",
)
def test_render_flowchart_svg_writes_to_output_path(tmp_path):
    target = tmp_path / "tree.svg"
    out = render_flowchart_svg(output_path=target)
    assert target.is_file()
    on_disk = target.read_text(encoding="utf-8")
    # Helper returns the same content it wrote
    assert out == on_disk
    assert "<svg" in on_disk.lower()
