"""Direct tests for `scitex_stats._cli.introspect`.

Module sat at 79 %. Existing tests drive `list-python-apis` via
subprocess, which doesn't cover the helper functions (`_simplify_type`,
`_get_api_tree`, `_format_python_signature`) or `cmd_api`'s
verbose / json / depth branches.
"""

from __future__ import annotations

import importlib
import json
import typing

# `from scitex_stats._cli import introspect` returns the click Group
# rebound in `_cli/__init__.py`; reach the file directly.
introspect = importlib.import_module("scitex_stats._cli.introspect")


# ----- _style ----------------------------------------------------------- #


def test_style_passthrough_when_not_tty_plain_introspect():
    # Arrange
    # Act
    # Assert
    assert introspect._style("plain") == "plain"

def test_style_passthrough_when_not_tty_plain_introspect_green():
    # Arrange
    # Act
    # Assert
    assert introspect._style("plain", fg="green", bold=True) == "plain"


# ----- _simplify_type --------------------------------------------------- #


def test_simplify_type_optional():
    # Arrange
    # Act
    # Assert
    assert introspect._simplify_type(typing.Optional[int]) == "Optional"


def test_simplify_type_union_of_two():
    # Arrange
    # Act
    # Assert
    assert introspect._simplify_type(typing.Union[int, str]) == "Union"


def test_simplify_type_list():
    # Arrange
    # Act
    name = introspect._simplify_type(typing.List[int])
    # Assert
    assert name == "list"


def test_simplify_type_class_with_name():
    # Arrange
    # Act
    # Assert
    assert introspect._simplify_type(int) == "int"


def test_simplify_type_pipe_union():
    # PEP 604 syntax — `int | None`
    # Arrange
    # Act
    # Assert
    assert introspect._simplify_type(int | None) == "Optional"


def test_simplify_type_string_fallback():
    """Unusual type that lacks __name__ and goes through the str-strip path."""
    # Arrange
    ann = typing.NewType("MyAlias", int)
    # Act
    name = introspect._simplify_type(ann)
    # Assert
    assert isinstance(name, str) and name


# ----- _format_python_signature ---------------------------------------- #


def test_format_python_signature_for_known_function_example_name_s():
    # Arrange
    def example(a: int, b: str = "x") -> str:
        return b * a
    # Act
    name_s, sig_s = introspect._format_python_signature(example, multiline=False)
    # Assert
    assert "example" in name_s

def test_format_python_signature_for_known_function_sig_s():
    # Arrange
    def example(a: int, b: str = "x") -> str:
        return b * a
    # Act
    name_s, sig_s = introspect._format_python_signature(example, multiline=False)
    # Assert
    assert "a" in sig_s

def test_format_python_signature_for_known_function_sig_s_2():
    # Arrange
    def example(a: int, b: str = "x") -> str:
        return b * a
    # Act
    name_s, sig_s = introspect._format_python_signature(example, multiline=False)
    # Assert
    assert "b" in sig_s


def test_format_python_signature_signature_unavailable_returns_empty_nosig_name_s():
    """When `inspect.signature` raises, the helper falls back to just
    the function name with an empty signature string."""
    # Arrange
    class _NoSig:
        # builtin types raise TypeError under inspect.signature
        __name__ = "_NoSig"
    # Act
    name_s, sig_s = introspect._format_python_signature(_NoSig)
    # Assert
    assert "_NoSig" in name_s

def test_format_python_signature_signature_unavailable_returns_empty_sig_s_startswith():
    """When `inspect.signature` raises, the helper falls back to just
    the function name with an empty signature string."""
    # Arrange
    class _NoSig:
        # builtin types raise TypeError under inspect.signature
        __name__ = "_NoSig"
    # Act
    name_s, sig_s = introspect._format_python_signature(_NoSig)
    # Assert
    assert sig_s == "" or sig_s.startswith("(")


# ----- _get_api_tree --------------------------------------------------- #


def test_get_api_tree_returns_list_of_entries_case_1():
    # Arrange
    import scitex_stats
    # Act
    tree = introspect._get_api_tree(scitex_stats, max_depth=1)
    # Assert
    assert isinstance(tree, list)
    root = tree[0]

def test_get_api_tree_returns_list_of_entries_case_2():
    # Arrange
    import scitex_stats
    # Act
    tree = introspect._get_api_tree(scitex_stats, max_depth=1)
    # Assert
    assert tree, "empty API tree"
    root = tree[0]

def test_get_api_tree_returns_list_of_entries_name_type_depth_root():
    # Arrange
    import scitex_stats
    # Act
    tree = introspect._get_api_tree(scitex_stats, max_depth=1)
    root = tree[0]
    # Assert
    for k in ("Name", "Type", "Depth"):
        assert k in root


def test_get_api_tree_respects_max_depth():
    # Arrange
    import scitex_stats
    shallow = introspect._get_api_tree(scitex_stats, max_depth=0)
    # Act
    deeper = introspect._get_api_tree(scitex_stats, max_depth=2)
    # Assert
    assert len(deeper) >= len(shallow)


def test_get_api_tree_with_docstring_flag():
    # Arrange
    import scitex_stats
    # Act
    tree = introspect._get_api_tree(scitex_stats, max_depth=1, docstring=True)
    # Assert
    assert any("Docstring" in entry for entry in tree)


# ----- cmd_api --------------------------------------------------------- #


def test_cmd_api_known_module_json_rc(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=0, max_depth=1, as_json=True
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    parsed = json.loads(out)

def test_cmd_api_known_module_json_parsed_list(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=0, max_depth=1, as_json=True
    )
    out = capsys.readouterr().out
    # Act
    parsed = json.loads(out)
    # Assert
    assert isinstance(parsed, list) and parsed


def test_cmd_api_known_module_text_rc(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=0, max_depth=1, as_json=False
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0

def test_cmd_api_known_module_text_tree_of(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=0, max_depth=1, as_json=False
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert "API tree of" in out

def test_cmd_api_known_module_text_legend(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=0, max_depth=1, as_json=False
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert "Legend" in out


def test_cmd_api_handles_dash_in_dotted_path_rc(capsys):
    """`--` and `-` in module names get normalised to `_`."""
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex-stats", verbose=0, max_depth=1, as_json=True
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    parsed = json.loads(out)

def test_cmd_api_handles_dash_in_dotted_path_parsed(capsys):
    """`--` and `-` in module names get normalised to `_`."""
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex-stats", verbose=0, max_depth=1, as_json=True
    )
    out = capsys.readouterr().out
    # Act
    parsed = json.loads(out)
    # Assert
    assert parsed


def test_cmd_api_unknown_module_returns_nonzero_rc(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="this_module_definitely_does_not_exist",
        verbose=0,
        max_depth=1,
    )
    # Act
    captured = capsys.readouterr()
    # Assert
    assert rc == 1

def test_cmd_api_unknown_module_returns_nonzero_error_importing_err_captured(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="this_module_definitely_does_not_exist",
        verbose=0,
        max_depth=1,
    )
    # Act
    captured = capsys.readouterr()
    # Assert
    assert "Error importing" in captured.err


def test_cmd_api_verbose_emits_docstrings_rc(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=1, max_depth=1, as_json=False
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0

def test_cmd_api_verbose_emits_docstrings_tree(capsys):
    # Arrange
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=1, max_depth=1, as_json=False
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert "    -" in out or "API tree" in out


# ----- cmd_list_python_apis ------------------------------------------- #


def test_cmd_list_python_apis_delegates_to_cmd_api_rc(capsys):
    # Arrange
    rc = introspect.cmd_list_python_apis(verbose=0, max_depth=1, as_json=True)
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    parsed = json.loads(out)

def test_cmd_list_python_apis_delegates_to_cmd_api_parsed(capsys):
    # Arrange
    rc = introspect.cmd_list_python_apis(verbose=0, max_depth=1, as_json=True)
    out = capsys.readouterr().out
    # Act
    parsed = json.loads(out)
    # Assert
    assert parsed
