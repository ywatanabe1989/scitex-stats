#!/usr/bin/env python3
# File: src/scitex_stats/_cli/introspect.py

"""Introspection CLI commands for scitex-stats."""

import argparse
import importlib
import inspect
import sys

TYPE_COLORS = {"M": "blue", "C": "magenta", "F": "green", "V": "cyan"}

ANSI = {
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def _style(text: str, fg: str = None, bold: bool = False) -> str:
    """Apply ANSI styling to text."""
    if not sys.stdout.isatty():
        return text
    prefix = ""
    if bold:
        prefix += ANSI["bold"]
    if fg and fg in ANSI:
        prefix += ANSI[fg]
    if prefix:
        return f"{prefix}{text}{ANSI['reset']}"
    return text


def _simplify_type(ann) -> str:
    """Simplify type annotation to base type name."""
    import types
    import typing

    if isinstance(ann, types.UnionType):
        args = typing.get_args(ann)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return "Optional"
        return "Union"

    origin = typing.get_origin(ann)
    if origin is not None:
        if origin is typing.Union:
            args = typing.get_args(ann)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1 and type(None) in args:
                return "Optional"
            return "Union"
        return origin.__name__ if hasattr(origin, "__name__") else str(origin)

    if hasattr(ann, "__name__"):
        return ann.__name__

    type_str = str(ann).replace("typing.", "")
    if "[" in type_str:
        type_str = type_str.split("[")[0]
    return type_str


def _format_python_signature(func, multiline: bool = True, indent: str = "  ") -> tuple:
    """Format Python function signature with colors."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return _style(func.__name__, "green", bold=True), ""

    params = []
    for name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            type_str = _simplify_type(param.annotation)
        else:
            type_str = None

        if param.default != inspect.Parameter.empty:
            default = param.default
            def_str = repr(default) if len(repr(default)) < 20 else "..."
            if type_str:
                p = f"{_style(name, 'white', bold=True)}: {_style(type_str, 'cyan')} = {_style(def_str, 'yellow')}"
            else:
                p = f"{_style(name, 'white', bold=True)} = {_style(def_str, 'yellow')}"
        else:
            if type_str:
                p = f"{_style(name, 'white', bold=True)}: {_style(type_str, 'cyan')}"
            else:
                p = _style(name, "white", bold=True)
        params.append(p)

    ret_str = ""
    if sig.return_annotation != inspect.Parameter.empty:
        ret = sig.return_annotation
        ret_name = ret.__name__ if hasattr(ret, "__name__") else str(ret)
        ret_name = ret_name.replace("typing.", "")
        ret_str = f" -> {_style(ret_name, 'magenta')}"

    name_s = _style(func.__name__, "green", bold=True)

    if multiline and len(params) > 2:
        param_indent = indent + "    "
        params_str = ",\n".join(f"{param_indent}{p}" for p in params)
        sig_s = f"(\n{params_str}\n{indent}){ret_str}"
    else:
        sig_s = f"({', '.join(params)}){ret_str}"

    return name_s, sig_s


def _get_api_tree(module, max_depth: int = 5, docstring: bool = False) -> list:
    """Get API tree for a module."""
    results = []

    def _visit(obj, name: str, depth: int, visited: set):
        if depth > max_depth:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if inspect.ismodule(obj):
            obj_type = "M"
        elif inspect.isclass(obj):
            obj_type = "C"
        elif callable(obj):
            obj_type = "F"
        else:
            obj_type = "V"

        entry = {"Name": name, "Type": obj_type, "Depth": depth}
        if docstring:
            entry["Docstring"] = inspect.getdoc(obj) or ""
        results.append(entry)

        if inspect.ismodule(obj) and depth < max_depth:
            if hasattr(obj, "__all__"):
                members = [(n, getattr(obj, n, None)) for n in obj.__all__]
            else:
                members = [
                    (n, v) for n, v in inspect.getmembers(obj) if not n.startswith("_")
                ]
            for member_name, member_obj in members:
                if member_obj is not None:
                    _visit(member_obj, f"{name}.{member_name}", depth + 1, visited)

    _visit(module, module.__name__.split(".")[-1], 0, set())
    return results


def cmd_api(args: argparse.Namespace) -> int:
    """List API tree of a Python module."""
    dotted_path = args.dotted_path.replace("-", "_")

    try:
        module = importlib.import_module(dotted_path)
    except ImportError as e:
        print(f"Error importing {dotted_path}: {e}", file=sys.stderr)
        return 1

    df = _get_api_tree(module, max_depth=args.max_depth, docstring=(args.verbose >= 1))

    if args.json:
        import json

        print(json.dumps(df, indent=2))
        return 0

    print(_style(f"API tree of {dotted_path} ({len(df)} items):", fg="cyan"))
    legend = " ".join(
        _style(f"[{t}]={n}", fg=TYPE_COLORS[t])
        for t, n in [
            ("M", "Module"),
            ("C", "Class"),
            ("F", "Function"),
            ("V", "Variable"),
        ]
    )
    print(f"Legend: {legend}")

    for row in df:
        indent = "  " * row["Depth"]
        t = row["Type"]
        type_s = _style(f"[{t}]", fg=TYPE_COLORS.get(t, "yellow"))
        name = row["Name"].split(".")[-1]

        if t == "F":
            try:
                parts = row["Name"].split(".")
                obj = module
                for part in parts[1:]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj and callable(obj):
                    name_s, sig_s = _format_python_signature(obj, indent=indent)
                    print(f"{indent}{type_s} {name_s}{sig_s}")
                else:
                    name_s = _style(name, "green", bold=True)
                    print(f"{indent}{type_s} {name_s}")
            except Exception:
                name_s = _style(name, "green", bold=True)
                print(f"{indent}{type_s} {name_s}")
        else:
            name_s = _style(name, fg=TYPE_COLORS.get(t, "white"), bold=True)
            print(f"{indent}{type_s} {name_s}")

        if args.verbose >= 1 and row.get("Docstring"):
            if args.verbose == 1:
                doc = row["Docstring"].split("\n")[0][:60]
                print(f"{indent}    - {doc}")
            else:
                for ln in row["Docstring"].split("\n"):
                    print(f"{indent}    {ln}")

    return 0


def cmd_list_python_apis(args: argparse.Namespace) -> int:
    """List Python APIs (alias for introspect api scitex_stats)."""
    args.dotted_path = "scitex_stats"
    return cmd_api(args)


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register `python-api` noun group (replaces verb-shaped `introspect`)."""
    intro_help = """Python package introspection — `python-api list` / `show`.

Quick start:
  scitex-stats python-api list scitex_stats       # Full API tree
  scitex-stats python-api list scitex_stats -v    # With docstrings
  scitex-stats python-api list scitex_stats --json  # JSON output
"""
    intro_parser = subparsers.add_parser(
        "python-api",
        help="Python package introspection (list API tree).",
        description=intro_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    intro_sub = intro_parser.add_subparsers(dest="python_api_command", title="Verbs")

    api_parser = intro_sub.add_parser(
        "list",
        help="List API tree of a Python module.",
        description=(
            "List the public API tree of a Python module (modules, classes,\n"
            "functions, variables) up to the requested depth.\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats python-api list scitex_stats\n"
            "  $ scitex-stats python-api list scitex_stats -v --max-depth 3\n"
            "  $ scitex-stats python-api list scitex_stats.correct --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    api_parser.add_argument(
        "dotted_path", help="Python dotted path (e.g., scitex_stats)"
    )
    api_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity: -v +doc, -vv full doc",
    )
    api_parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=5,
        help="Max recursion depth (default: 5)",
    )
    api_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON",
    )
    api_parser.set_defaults(func=cmd_api)

    return intro_parser


def register_list_python_apis(parent_parser) -> None:
    """Register list-python-apis convenience alias on a parent parser."""
    lst_parser = parent_parser.add_parser(
        "list-python-apis",
        help="List Python APIs (alias for: scitex-stats python-api list scitex_stats)",
        description=(
            "List the public API tree of scitex_stats (alias for `python-api list`).\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats list-python-apis\n"
            "  $ scitex-stats list-python-apis -v --max-depth 3\n"
            "  $ scitex-stats list-python-apis --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    lst_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity: -v +doc, -vv full doc",
    )
    lst_parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=5,
        help="Max recursion depth",
    )
    lst_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON",
    )
    lst_parser.set_defaults(func=cmd_list_python_apis)


# EOF
