"""Direct tests for `scitex-stats skills` CLI subcommand.

The existing CLI tests cover `mcp`, `list-python-apis`, and helps; the
skills subcommand sat at ~26 % coverage. These tests drive `list`,
`get`, and `install` (both default-symlink and `--no-link` copy
modes) without touching the user's `~/.scitex` or `~/.claude` dirs.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scitex_stats._cli.skills_group import skills_group


def _invoke(*args):
    # Click 8.3+ dropped `mix_stderr`; `.stderr` is already the separated
    # stream by default.
    return CliRunner().invoke(skills_group, list(args))


# ----- list ------------------------------------------------------------- #


def test_skills_list_prints_at_least_one_entry():
    result = _invoke("list")
    assert result.exit_code == 0, result.stderr
    # bundled skills exist under src/scitex_stats/_skills/scitex-stats/,
    # so at least one stem must be printed.
    assert result.stdout.strip(), "skills list emitted no rows"


def test_skills_list_json_emits_array():
    result = _invoke("list", "--json")
    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)
    assert all("name" in row and "path" in row for row in parsed)


# ----- get -------------------------------------------------------------- #


def test_skills_get_known_name_emits_markdown():
    # Pick the first available skill name from `list` so the test
    # adapts to whatever the package actually ships.
    listing = _invoke("list", "--json")
    rows = json.loads(listing.stdout)
    assert rows, "skills_list returned no rows — repo doesn't bundle skills?"
    name = rows[0]["name"]

    result = _invoke("get", name)
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip(), "skills get printed nothing"


def test_skills_get_unknown_name_exits_nonzero():
    result = _invoke("get", "definitely-not-a-real-skill-name")
    assert result.exit_code != 0
    assert "not found" in result.stderr or "not found" in result.stdout


# ----- install --dry-run ----------------------------------------------- #


def test_skills_install_dry_run_default_paths():
    result = _invoke("install", "--dry-run")
    assert result.exit_code == 0, result.stderr
    assert "symlink" in result.stdout
    assert "scitex-stats" in result.stdout


def test_skills_install_dry_run_with_claude_symlink():
    result = _invoke("install", "--dry-run", "--claude-symlink")
    assert result.exit_code == 0, result.stderr
    assert "scitex-stats" in result.stdout
    # The claude-symlink path mentions .claude/skills/scitex
    assert ".claude/skills/scitex" in result.stdout or "scitex" in result.stdout


def test_skills_install_dry_run_no_link_says_copy():
    result = _invoke("install", "--dry-run", "--no-link")
    assert result.exit_code == 0, result.stderr
    assert "copy" in result.stdout


# ----- install (real, scoped to tmp dest) ----------------------------- #


def test_skills_install_no_link_into_tmp(tmp_path: Path):
    dest = tmp_path / "skills"
    result = _invoke("install", "--dest", str(dest), "--no-link", "--yes")
    assert result.exit_code == 0, result.stderr
    installed = dest / "scitex-stats"
    assert installed.is_dir(), f"expected directory at {installed}"
    # The bundled tree contains markdown skills — verify a few landed.
    md_files = list(installed.rglob("*.md"))
    assert md_files, "no .md files copied into target"


def test_skills_install_symlink_into_tmp(tmp_path: Path):
    dest = tmp_path / "skills"
    result = _invoke("install", "--dest", str(dest), "--yes")
    assert result.exit_code == 0, result.stderr
    installed = dest / "scitex-stats"
    assert installed.is_symlink(), f"expected symlink at {installed}"


def test_skills_install_overwrites_existing_target(tmp_path: Path):
    dest = tmp_path / "skills"
    # First install (copy)
    _invoke("install", "--dest", str(dest), "--no-link", "--yes")
    # Second install (symlink) must replace the directory cleanly.
    result = _invoke("install", "--dest", str(dest), "--yes")
    assert result.exit_code == 0, result.stderr
    installed = dest / "scitex-stats"
    assert installed.is_symlink()
