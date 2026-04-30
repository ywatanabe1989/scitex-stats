#!/usr/bin/env python3
# File: src/scitex_stats/_cli/__init__.py

"""CLI package for scitex-stats.

Subcommands:
    mcp              - MCP server commands
    list-python-apis - List Python APIs
    introspect       - Python package introspection
"""

import argparse
import sys

from .. import __version__
from . import introspect, mcp, stats


def _cmd_help_recursive(parser: argparse.ArgumentParser) -> int:
    """Show help for all commands recursively."""
    print("=" * 60)
    print("SciTeX Stats - Complete Command Reference")
    print("=" * 60)
    print()

    parser.print_help()
    print()

    for action in parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in sorted(action.choices.items()):
                print("-" * 60)
                print(f"Command: {name}")
                print("-" * 60)
                subparser.print_help()
                print()

                # Recurse into sub-subparsers
                if subparser._subparsers is None:
                    continue
                for sub_action in subparser._subparsers._actions:
                    if isinstance(sub_action, argparse._SubParsersAction):
                        for sub_name, sub_subparser in sorted(
                            sub_action.choices.items()
                        ):
                            print(f"  {'.' * 56}")
                            print(f"  Subcommand: {name} {sub_name}")
                            print(f"  {'.' * 56}")
                            sub_subparser.print_help()
                            print()

    return 0


def main(argv=None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="scitex-stats",
        description=(
            "SciTeX Stats - Publication-ready statistical testing framework "
            "with 23 tests, effect sizes, power analysis, and MCP server"
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--help-recursive",
        action="store_true",
        help="Show help for all commands recursively",
    )

    subparsers = parser.add_subparsers(dest="command", title="Commands")

    # Register subcommand modules
    mcp_parser = mcp.register_parser(subparsers)
    introspect.register_parser(subparsers)
    stats.register_parsers(subparsers)

    # Register top-level convenience commands
    introspect.register_list_python_apis(subparsers)

    # Docs and Skills subcommands (from scitex-dev)
    try:
        from scitex_dev.cli import register_docs_subcommand, register_skills_subcommand

        register_docs_subcommand(subparsers, package="scitex-stats")
        register_skills_subcommand(subparsers, package="scitex-stats")
    except ImportError:
        pass

    args = parser.parse_args(argv)

    # Handle --help-recursive
    if args.help_recursive:
        return _cmd_help_recursive(parser)

    # Handle command dispatch
    if hasattr(args, "func"):
        return args.func(args)

    # No subcommand - show help for command group
    parsers = {
        "mcp": mcp_parser,
    }

    if args.command in parsers:
        parsers[args.command].print_help()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# EOF
