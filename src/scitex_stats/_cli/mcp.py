#!/usr/bin/env python3
# File: src/scitex_stats/_cli/mcp.py

"""MCP CLI commands for scitex-stats."""

import argparse
import shutil
import sys

from .. import __version__

CLAUDE_DESKTOP_CONFIG_CLI = """{
  "mcpServers": {
    "scitex-stats": {
      "command": "/path/to/.venv/bin/scitex-stats",
      "args": ["mcp", "start"]
    }
  }
}"""

CLAUDE_DESKTOP_CONFIG_PYTHON = """{
  "mcpServers": {
    "scitex-stats": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "scitex_stats", "mcp", "start"]
    }
  }
}"""


def cmd_start(args: argparse.Namespace) -> int:
    """Start the MCP server."""
    if getattr(args, "dry_run", False):
        print(
            f"DRY RUN — would start scitex-stats MCP server (transport={args.transport})"
        )
        return 0
    if not getattr(args, "yes", False) and not sys.stdin.isatty():
        # Non-interactive context without --yes — proceed (no prompt to wait on)
        pass
    from .._mcp import run_server

    run_server(transport=args.transport)
    return 0


def _style(text: str, fg: str = None, bold: bool = False) -> str:
    """Apply ANSI color styling."""
    import sys

    if not sys.stdout.isatty():
        return text
    codes = {
        "green": "\033[32m",
        "cyan": "\033[36m",
        "yellow": "\033[33m",
        "magenta": "\033[35m",
        "white": "\033[37m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    prefix = ""
    if bold:
        prefix += codes["bold"]
    if fg and fg in codes:
        prefix += codes[fg]
    return f"{prefix}{text}{codes['reset']}" if prefix else text


def _format_tool_signature(tool, compact: bool = False, indent: str = "  ") -> str:
    """Format tool as Python-like function signature with colors."""
    import inspect

    params = []
    if hasattr(tool, "parameters") and tool.parameters:
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for name, info in props.items():
            ptype = info.get("type", "any")
            default = info.get("default")
            name_s = _style(name, "white", bold=True)
            type_s = _style(ptype, "cyan")
            if name in required:
                params.append(f"{name_s}: {type_s}")
            elif default is not None:
                def_str = repr(default) if len(repr(default)) < 20 else "..."
                def_s = _style(f"= {def_str}", "yellow")
                params.append(f"{name_s}: {type_s} {def_s}")
            else:
                def_s = _style("= None", "yellow")
                params.append(f"{name_s}: {type_s} {def_s}")

    ret_type = ""
    if hasattr(tool, "fn") and tool.fn:
        try:
            sig = inspect.signature(tool.fn)
            if sig.return_annotation != inspect.Parameter.empty:
                ret = sig.return_annotation
                ret_name = ret.__name__ if hasattr(ret, "__name__") else str(ret)
                ret_type = f" -> {_style(ret_name, 'magenta')}"
        except Exception:
            pass

    name_s = _style(tool.name, "green")
    if compact or len(params) <= 2:
        return f"{indent}{name_s}({', '.join(params)}){ret_type}"
    else:
        param_indent = indent + "    "
        params_str = ",\n".join(f"{param_indent}{p}" for p in params)
        return f"{indent}{name_s}(\n{params_str}\n{indent}){ret_type}"


def _get_tool_module(name: str) -> str:
    """Get logical module for a tool name."""
    if "recommend" in name:
        return "auto"
    if "correct" in name:
        return "correct"
    if "posthoc" in name:
        return "posthoc"
    if "power" in name:
        return "power"
    if "effect" in name:
        return "effect_sizes"
    if "normality" in name:
        return "normality"
    if "describe" in name:
        return "descriptive"
    if "format" in name:
        return "formatting"
    if "p_to_stars" in name:
        return "formatting"
    return "general"


def cmd_list_tools(args: argparse.Namespace) -> int:
    """List all available MCP tools."""
    import asyncio

    from .._mcp import mcp

    verbose = getattr(args, "verbose", 0)
    compact = getattr(args, "compact", False)
    module_filter = getattr(args, "module", None)
    as_json = getattr(args, "json", False)

    tool_objects = asyncio.run(mcp.list_tools())
    tools_by_name = {t.name: t for t in tool_objects}
    tools = list(tools_by_name.keys())
    total = len(tools)

    # Group by logical module
    modules = {}
    for tool_name in sorted(tools):
        module = _get_tool_module(tool_name)
        if module not in modules:
            modules[module] = []
        modules[module].append(tool_name)

    if module_filter:
        module_filter = module_filter.lower()
        if module_filter not in modules:
            print(f"ERROR: Unknown module '{module_filter}'")
            print(f"Available modules: {', '.join(sorted(modules.keys()))}")
            return 1
        modules = {module_filter: modules[module_filter]}

    if as_json:
        import json

        output = {
            "name": "scitex-stats",
            "total": sum(len(t) for t in modules.values()),
            "modules": {},
        }
        for mod, tool_list in modules.items():
            output["modules"][mod] = {
                "count": len(tool_list),
                "tools": tool_list,
            }
        print(json.dumps(output, indent=2))
        return 0

    print(_style("SciTeX Stats MCP: scitex-stats", "cyan", bold=True))
    print(f"Tools: {total} ({len(modules)} modules)\n")

    for module in sorted(modules.keys()):
        mod_tools = sorted(modules[module])
        print(_style(f"{module}: {len(mod_tools)} tools", "green", bold=True))
        for tool_name in mod_tools:
            tool_obj = tools_by_name.get(tool_name)

            if verbose == 0:
                print(f"  {tool_name}")
            elif verbose == 1:
                sig = (
                    _format_tool_signature(tool_obj, compact=compact)
                    if tool_obj
                    else f"  {tool_name}"
                )
                print(sig)
            elif verbose == 2:
                sig = (
                    _format_tool_signature(tool_obj, compact=compact)
                    if tool_obj
                    else f"  {tool_name}"
                )
                print(sig)
                if tool_obj and tool_obj.description:
                    desc = tool_obj.description.split("\n")[0].strip()
                    print(f"    {desc}")
                print()
            else:
                sig = (
                    _format_tool_signature(tool_obj, compact=compact)
                    if tool_obj
                    else f"  {tool_name}"
                )
                print(sig)
                if tool_obj and tool_obj.description:
                    for line in tool_obj.description.strip().split("\n"):
                        print(f"    {line}")
                print()
        print()

    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check MCP server health and configuration."""
    print(f"scitex-stats {__version__}\n")
    print("Health Check")
    print("=" * 40)

    checks = []

    try:
        import fastmcp

        checks.append(("fastmcp", True, fastmcp.__version__))
    except ImportError:
        checks.append(("fastmcp", False, "not installed"))

    try:
        import asyncio

        from .._mcp import mcp

        tool_count = len(asyncio.run(mcp.list_tools()))
        checks.append(("MCP server", True, f"{tool_count} tools"))
    except Exception as e:
        checks.append(("MCP server", False, str(e)))

    scitex_path = shutil.which("scitex-stats")
    if scitex_path:
        checks.append(("CLI", True, scitex_path))
    else:
        checks.append(("CLI", False, "not in PATH"))

    all_ok = True
    for name, ok, info in checks:
        status = "+" if ok else "x"
        if not ok:
            all_ok = False
        print(f"  {status} {name}: {info}")

    print()
    if all_ok:
        print("All checks passed!")
    else:
        print("Some checks failed. Run 'pip install scitex-stats[mcp]' to fix.")

    return 0 if all_ok else 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show Claude Desktop configuration snippet."""
    scitex_path = shutil.which("scitex-stats")

    if getattr(args, "json", False):
        import json as _json

        payload = {
            "package": "scitex-stats",
            "version": __version__,
            "installation_path": scitex_path,
            "config_paths": {
                "macos": "~/Library/Application Support/Claude/claude_desktop_config.json",
                "linux": "~/.config/Claude/claude_desktop_config.json",
            },
            "snippets": {
                "cli": CLAUDE_DESKTOP_CONFIG_CLI,
                "python_module": CLAUDE_DESKTOP_CONFIG_PYTHON,
            },
        }
        print(_json.dumps(payload, indent=2))
        return 0

    print(f"scitex-stats {__version__}\n")
    print("Add this to your Claude Desktop config file:\n")
    print("  macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("  Linux: ~/.config/Claude/claude_desktop_config.json\n")

    if scitex_path:
        print(f"Your installation path: {scitex_path}\n")

    print("Option 1: CLI command (replace path with your installation)")
    print(CLAUDE_DESKTOP_CONFIG_CLI)
    print("\nOption 2: Python module (replace path with your installation)")
    print(CLAUDE_DESKTOP_CONFIG_PYTHON)
    return 0


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register MCP subcommand parser."""
    mcp_help = """MCP (Model Context Protocol) server commands.

Quick start:
  scitex-stats mcp list-tools         # List all tools
  scitex-stats mcp doctor             # Check server health
  scitex-stats mcp show-installation  # Print Claude Desktop installation snippet
  scitex-stats mcp start              # Start MCP server
"""
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="MCP server commands",
        description=mcp_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_command", title="Commands")

    inst = mcp_sub.add_parser(
        "show-installation",
        help="Show Claude Desktop installation guide",
        description=(
            "Print the Claude Desktop config snippet for scitex-stats MCP server.\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats mcp show-installation\n"
            "  $ scitex-stats mcp show-installation --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inst.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of pretty text.",
    )
    inst.set_defaults(func=cmd_config)

    lst = mcp_sub.add_parser(
        "list-tools",
        help="List all available MCP tools",
        description=(
            "List all MCP tools registered under scitex-stats.\n"
            "Verbosity: (none) names, -v signatures, -vv +description, -vvv full.\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats mcp list-tools\n"
            "  $ scitex-stats mcp list-tools -vv\n"
            "  $ scitex-stats mcp list-tools --module correct --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    lst.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity: -v sig, -vv +desc, -vvv full",
    )
    lst.add_argument(
        "-c", "--compact", action="store_true", help="Compact signatures (single line)"
    )
    lst.add_argument(
        "-m",
        "--module",
        type=str,
        default=None,
        help="Filter by module (auto, correct, descriptive, effect_sizes, formatting, general, normality, posthoc, power)",
    )
    lst.add_argument(
        "--json", action="store_true", default=False, help="Output as JSON"
    )
    lst.set_defaults(func=cmd_list_tools)

    doc = mcp_sub.add_parser(
        "doctor",
        help="Check MCP server health",
        description=(
            "Verify scitex-stats MCP server dependencies + tool count + CLI presence.\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats mcp doctor"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doc.set_defaults(func=cmd_doctor)

    start = mcp_sub.add_parser(
        "start",
        help="Start the MCP server",
        description=(
            "Launch the scitex-stats MCP server (stdio or SSE transport).\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats mcp start                  # stdio (default)\n"
            "  $ scitex-stats mcp start --transport sse  # SSE for HTTP clients\n"
            "  $ scitex-stats mcp start --dry-run        # show what would launch"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    start.add_argument("-t", "--transport", choices=["stdio", "sse"], default="stdio")
    start.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch plan without starting the server.",
    )
    start.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Suppress interactive confirmation (assume yes).",
    )
    start.set_defaults(func=cmd_start)

    return mcp_parser


# EOF
