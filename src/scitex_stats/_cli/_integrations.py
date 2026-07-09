#!/usr/bin/env python3
# File: src/scitex_stats/_cli/_integrations.py
"""Optional scitex-dev integrations wired onto the root Click group.

Split out of ``_cli/__init__.py`` to keep that module under the repo's
512-line file-size limit. Both integrations are best-effort: scitex-dev
being absent, or older than expected, must never break scitex-stats's own
CLI.
"""

from __future__ import annotations


def attach_scitex_dev_integrations(main) -> None:
    """Attach shell-completion leaves + the optional docs subcommand.

    Called once, at import time, from ``_cli/__init__.py`` after ``main``
    and all of its own subcommands/groups are fully defined.
    """
    # §1a: install-shell-completion + print-shell-completion (canonical leaves)
    try:
        from scitex_dev._cli._completion import attach_shell_completion

        attach_shell_completion(main, prog_name="scitex-stats")
    except ImportError:
        pass

    # Optional docs/skills subcommands from scitex-dev. scitex-dev exposes
    # argparse-compatible registration functions; only wire in the Click
    # variant when available. Otherwise skip silently — the docs/skills
    # subcommands are non-essential and the canonical API is `scitex-dev`.
    try:
        from scitex_dev.cli import (  # noqa: F401
            register_docs_subcommand,
            register_skills_subcommand,
        )

        try:
            from scitex_dev.cli import (  # type: ignore
                register_docs_click_command,
                register_skills_click_command,  # noqa: F401
            )
        except ImportError:
            return

        register_docs_click_command(main, package="scitex-stats")
        # Skills group is owned locally (skills_group.py) — do not override.
    except ImportError:
        pass


# EOF
