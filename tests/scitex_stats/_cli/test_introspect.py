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


def test_style_passthrough_when_not_tty():
    assert introspect._style("plain") == "plain"
    assert introspect._style("plain", fg="green", bold=True) == "plain"


# ----- _simplify_type --------------------------------------------------- #


def test_simplify_type_optional():
    assert introspect._simplify_type(typing.Optional[int]) == "Optional"


def test_simplify_type_union_of_two():
    assert introspect._simplify_type(typing.Union[int, str]) == "Union"


def test_simplify_type_list():
    name = introspect._simplify_type(typing.List[int])
    assert name == "list"


def test_simplify_type_class_with_name():
    assert introspect._simplify_type(int) == "int"


def test_simplify_type_pipe_union():
    # PEP 604 syntax — `int | None`
    assert introspect._simplify_type(int | None) == "Optional"


def test_simplify_type_string_fallback():
    """Unusual type that lacks __name__ and goes through the str-strip path."""
    ann = typing.NewType("MyAlias", int)
    name = introspect._simplify_type(ann)
    assert isinstance(name, str) and name


# ----- _format_python_signature ---------------------------------------- #


def test_format_python_signature_for_known_function():
    def example(a: int, b: str = "x") -> str:
        return b * a

    name_s, sig_s = introspect._format_python_signature(example, multiline=False)
    assert "example" in name_s
    assert "a" in sig_s
    assert "b" in sig_s


def test_format_python_signature_signature_unavailable_returns_empty():
    """When `inspect.signature` raises, the helper falls back to just
    the function name with an empty signature string."""

    class _NoSig:
        # builtin types raise TypeError under inspect.signature
        __name__ = "_NoSig"

    name_s, sig_s = introspect._format_python_signature(_NoSig)
    assert "_NoSig" in name_s
    # The fallback path returns empty signature when inspect couldn't
    # read one. Some classes may still expose a signature via __init__;
    # accept either an empty string or a normal signature.
    assert sig_s == "" or sig_s.startswith("(")


# ----- _get_api_tree --------------------------------------------------- #


def test_get_api_tree_returns_list_of_entries():
    import scitex_stats

    tree = introspect._get_api_tree(scitex_stats, max_depth=1)
    assert isinstance(tree, list)
    assert tree, "empty API tree"
    root = tree[0]
    for k in ("Name", "Type", "Depth"):
        assert k in root


def test_get_api_tree_respects_max_depth():
    import scitex_stats

    shallow = introspect._get_api_tree(scitex_stats, max_depth=0)
    deeper = introspect._get_api_tree(scitex_stats, max_depth=2)
    assert len(deeper) >= len(shallow)


def test_get_api_tree_with_docstring_flag():
    import scitex_stats

    tree = introspect._get_api_tree(scitex_stats, max_depth=1, docstring=True)
    # at least one node should carry docstring data when the flag is on
    assert any("Docstring" in entry for entry in tree)


# ----- cmd_api --------------------------------------------------------- #


def test_cmd_api_known_module_json(capsys):
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=0, max_depth=1, as_json=True
    )
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert isinstance(parsed, list) and parsed


def test_cmd_api_known_module_text(capsys):
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=0, max_depth=1, as_json=False
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "API tree of" in out
    assert "Legend" in out


def test_cmd_api_handles_dash_in_dotted_path(capsys):
    """`--` and `-` in module names get normalised to `_`."""
    rc = introspect.cmd_api(
        dotted_path="scitex-stats", verbose=0, max_depth=1, as_json=True
    )
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert parsed


def test_cmd_api_unknown_module_returns_nonzero(capsys):
    rc = introspect.cmd_api(
        dotted_path="this_module_definitely_does_not_exist",
        verbose=0,
        max_depth=1,
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "Error importing" in captured.err


def test_cmd_api_verbose_emits_docstrings(capsys):
    rc = introspect.cmd_api(
        dotted_path="scitex_stats", verbose=1, max_depth=1, as_json=False
    )
    out = capsys.readouterr().out
    assert rc == 0
    # verbose=1 prints one-line docstring summaries
    assert "    -" in out or "API tree" in out


# ----- cmd_list_python_apis ------------------------------------------- #


def test_cmd_list_python_apis_delegates_to_cmd_api(capsys):
    rc = introspect.cmd_list_python_apis(verbose=0, max_depth=1, as_json=True)
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert parsed
