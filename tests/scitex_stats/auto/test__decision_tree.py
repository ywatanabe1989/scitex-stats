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


def test_decision_tree_dict_shape_case_1():
    # Arrange
    # Act
    out = get_decision_tree()
    # Assert
    assert isinstance(out, dict) and out
    sample = next(iter(out.values()))

def test_decision_tree_dict_shape_id_label_emphasis_children():
    # Arrange
    out = get_decision_tree()
    # Act
    sample = next(iter(out.values()))
    # Assert
    for k in ("id", "label", "shape", "emphasis", "children", "test_id"):
        assert k in sample


def test_decision_tree_module_dict_consistent_with_helper():
    # Arrange
    # Act
    helper = get_decision_tree()
    # Assert
    assert set(helper.keys()) == set(DECISION_TREE.keys())


def test_decision_node_dataclass_fields_round_trip_decisionnode():
    # Arrange
    # Act
    sample_id = next(iter(DECISION_TREE))
    node = DECISION_TREE[sample_id]
    # Assert
    assert isinstance(node, DecisionNode)

def test_decision_node_dataclass_fields_round_trip_children_edge_label_child_id_str():
    # Arrange
    # Act
    sample_id = next(iter(DECISION_TREE))
    node = DECISION_TREE[sample_id]
    # Assert
    for edge_label, child_id in node.children:
        assert isinstance(edge_label, str) and isinstance(child_id, str)


def test_get_leaf_node_test_ids_only_returns_test_carrying_nodes_leaves_dict():
    # Arrange
    # Act
    leaves = get_leaf_node_test_ids()
    # Assert
    assert isinstance(leaves, dict) and leaves

def test_get_leaf_node_test_ids_only_returns_test_carrying_nodes_node_id_test_id_items_leaves():
    # Arrange
    # Act
    leaves = get_leaf_node_test_ids()
    # Assert
    for node_id, test_id in leaves.items():
        assert test_id is not None and DECISION_TREE[node_id].test_id == test_id


def test_render_flowchart_mermaid_emits_non_empty_markup_str_strip():
    # Arrange
    # Act
    out = render_flowchart_mermaid()
    # Assert
    assert isinstance(out, str) and out.strip()

def test_render_flowchart_mermaid_emits_non_empty_markup_graph_lower():
    # Arrange
    # Act
    out = render_flowchart_mermaid()
    # Assert
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
def test_render_flowchart_svg_returns_svg_string_str_strip():
    # Arrange
    # Act
    out = render_flowchart_svg()
    # Assert
    assert isinstance(out, str) and out.strip()

@pytest.mark.skipif(
    not _MMDC_OK,
    reason="render_flowchart_svg needs a working mmdc + Chromium chain",
)
def test_render_flowchart_svg_returns_svg_string_lower():
    # Arrange
    # Act
    out = render_flowchart_svg()
    # Assert
    assert "<svg" in out.lower()


@pytest.mark.skipif(
    not _MMDC_OK,
    reason="render_flowchart_svg needs a working mmdc + Chromium chain",
)
def test_render_flowchart_svg_writes_to_output_path_is_file_target(tmp_path):
    # Arrange
    target = tmp_path / "tree.svg"
    # Act
    out = render_flowchart_svg(output_path=target)
    # Assert
    assert target.is_file()
    on_disk = target.read_text(encoding="utf-8")

@pytest.mark.skipif(
    not _MMDC_OK,
    reason="render_flowchart_svg needs a working mmdc + Chromium chain",
)
def test_render_flowchart_svg_writes_to_output_path_on_disk(tmp_path):
    # Arrange
    target = tmp_path / "tree.svg"
    out = render_flowchart_svg(output_path=target)
    # Act
    on_disk = target.read_text(encoding="utf-8")
    # Assert
    assert out == on_disk

@pytest.mark.skipif(
    not _MMDC_OK,
    reason="render_flowchart_svg needs a working mmdc + Chromium chain",
)
def test_render_flowchart_svg_writes_to_output_path_lower_on_disk(tmp_path):
    # Arrange
    target = tmp_path / "tree.svg"
    out = render_flowchart_svg(output_path=target)
    # Act
    on_disk = target.read_text(encoding="utf-8")
    # Assert
    assert "<svg" in on_disk.lower()
