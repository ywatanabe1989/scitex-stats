#!/usr/bin/env python3
# File: src/scitex_stats/_cli/_help.py
"""``--help-recursive`` callback: print help for the root and every subcommand.

Split out of ``_cli/__init__.py`` to keep that module under the repo's
512-line file-size limit.
"""

from __future__ import annotations

import click


def print_help_recursive(ctx: click.Context, _param, value):
    """Click eager-option callback for ``--help-recursive``."""
    if not value or ctx.resilient_parsing:
        return
    cmd = ctx.command
    click.echo(cmd.get_help(ctx))

    def _walk(c, parent_ctx, prefix):
        if isinstance(c, click.Group):
            for name in sorted(c.commands):
                sub = c.commands[name]
                sub_ctx = click.Context(sub, info_name=name, parent=parent_ctx)
                click.echo("\n---\n")
                click.echo(f"Command: {prefix}{name}\n")
                click.echo(sub.get_help(sub_ctx))
                _walk(sub, sub_ctx, f"{prefix}{name} ")

    _walk(cmd, ctx, "")
    ctx.exit(0)


# EOF
