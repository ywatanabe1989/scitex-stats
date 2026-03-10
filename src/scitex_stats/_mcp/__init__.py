#!/usr/bin/env python3
# File: src/scitex_stats/_mcp/__init__.py

"""SciTeX Stats MCP module.

Lazy re-exports from _server.py to avoid importing fastmcp
when only utility functions are needed.
"""


def __getattr__(name):
    if name in ("mcp", "run_server"):
        from scitex_stats._server import mcp, run_server

        globals()["mcp"] = mcp
        globals()["run_server"] = run_server
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "mcp",
    "run_server",
]

# EOF
