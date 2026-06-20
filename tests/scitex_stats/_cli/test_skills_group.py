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


def test_skills_list_prints_at_least_one_entry_exit_code():
    # Arrange
    # Act
    result = _invoke("list")
    # Assert
    assert result.exit_code == 0, result.stderr

def test_skills_list_prints_at_least_one_entry_strip_stdout():
    # Arrange
    # Act
    result = _invoke("list")
    # Assert
    assert result.stdout.strip(), "skills list emitted no rows"


def test_skills_list_json_emits_array_exit_code():
    # Arrange
    # Act
    result = _invoke("list", "--json")
    # Assert
    assert result.exit_code == 0, result.stderr
    parsed = json.loads(result.stdout)

def test_skills_list_json_emits_array_parsed():
    # Arrange
    result = _invoke("list", "--json")
    # Act
    parsed = json.loads(result.stdout)
    # Assert
    assert isinstance(parsed, list)

def test_skills_list_json_emits_array_all_row_parsed_name():
    # Arrange
    result = _invoke("list", "--json")
    # Act
    parsed = json.loads(result.stdout)
    # Assert
    assert all("name" in row and "path" in row for row in parsed)


# ----- get -------------------------------------------------------------- #


def test_skills_get_known_name_emits_markdown_rows():
    # Pick the first available skill name from `list` so the test
    # adapts to whatever the package actually ships.
    # Arrange
    listing = _invoke("list", "--json")
    # Act
    rows = json.loads(listing.stdout)
    # Assert
    assert rows, "skills_list returned no rows — repo doesn't bundle skills?"
    name = rows[0]["name"]
    result = _invoke("get", name)

def test_skills_get_known_name_emits_markdown_exit_code():
    # Pick the first available skill name from `list` so the test
    # adapts to whatever the package actually ships.
    # Arrange
    listing = _invoke("list", "--json")
    rows = json.loads(listing.stdout)
    name = rows[0]["name"]
    # Act
    result = _invoke("get", name)
    # Assert
    assert result.exit_code == 0, result.stderr

def test_skills_get_known_name_emits_markdown_strip_stdout():
    # Pick the first available skill name from `list` so the test
    # adapts to whatever the package actually ships.
    # Arrange
    listing = _invoke("list", "--json")
    rows = json.loads(listing.stdout)
    name = rows[0]["name"]
    # Act
    result = _invoke("get", name)
    # Assert
    assert result.stdout.strip(), "skills get printed nothing"


def test_skills_get_unknown_name_exits_nonzero_exit_code():
    # Arrange
    # Act
    result = _invoke("get", "definitely-not-a-real-skill-name")
    # Assert
    assert result.exit_code != 0

def test_skills_get_unknown_name_exits_nonzero_not_found_stderr_stdout():
    # Arrange
    # Act
    result = _invoke("get", "definitely-not-a-real-skill-name")
    # Assert
    assert "not found" in result.stderr or "not found" in result.stdout


# ----- install --dry-run ----------------------------------------------- #


def test_skills_install_dry_run_default_paths_exit_code():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run")
    # Assert
    assert result.exit_code == 0, result.stderr

def test_skills_install_dry_run_default_paths_symlink_stdout():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run")
    # Assert
    assert "symlink" in result.stdout

def test_skills_install_dry_run_default_paths_scitex_stats_stdout():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run")
    # Assert
    assert "scitex-stats" in result.stdout


def test_skills_install_dry_run_with_claude_symlink_exit_code():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run", "--claude-symlink")
    # Assert
    assert result.exit_code == 0, result.stderr

def test_skills_install_dry_run_with_claude_symlink_scitex_stats_stdout():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run", "--claude-symlink")
    # Assert
    assert "scitex-stats" in result.stdout

def test_skills_install_dry_run_with_claude_symlink_scitex_stdout():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run", "--claude-symlink")
    # Assert
    assert ".claude/skills/scitex" in result.stdout or "scitex" in result.stdout


def test_skills_install_dry_run_no_link_says_copy_exit_code():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run", "--no-link")
    # Assert
    assert result.exit_code == 0, result.stderr

def test_skills_install_dry_run_no_link_says_copy_stdout():
    # Arrange
    # Act
    result = _invoke("install", "--dry-run", "--no-link")
    # Assert
    assert "copy" in result.stdout


# ----- install (real, scoped to tmp dest) ----------------------------- #


def test_skills_install_no_link_into_tmp_exit_code(tmp_path: Path):
    # Arrange
    dest = tmp_path / "skills"
    # Act
    result = _invoke("install", "--dest", str(dest), "--no-link", "--yes")
    # Assert
    assert result.exit_code == 0, result.stderr
    installed = dest / "scitex-stats"
    md_files = list(installed.rglob("*.md"))

def test_skills_install_no_link_into_tmp_is_dir_installed(tmp_path: Path):
    # Arrange
    dest = tmp_path / "skills"
    # Act
    result = _invoke("install", "--dest", str(dest), "--no-link", "--yes")
    installed = dest / "scitex-stats"
    # Assert
    assert installed.is_dir(), f"expected directory at {installed}"
    md_files = list(installed.rglob("*.md"))

def test_skills_install_no_link_into_tmp_md_files(tmp_path: Path):
    # Arrange
    dest = tmp_path / "skills"
    result = _invoke("install", "--dest", str(dest), "--no-link", "--yes")
    installed = dest / "scitex-stats"
    # Act
    md_files = list(installed.rglob("*.md"))
    # Assert
    assert md_files, "no .md files copied into target"


def test_skills_install_symlink_into_tmp_exit_code(tmp_path: Path):
    # Arrange
    dest = tmp_path / "skills"
    # Act
    result = _invoke("install", "--dest", str(dest), "--yes")
    # Assert
    assert result.exit_code == 0, result.stderr
    installed = dest / "scitex-stats"

def test_skills_install_symlink_into_tmp_is_symlink_installed(tmp_path: Path):
    # Arrange
    dest = tmp_path / "skills"
    # Act
    result = _invoke("install", "--dest", str(dest), "--yes")
    installed = dest / "scitex-stats"
    # Assert
    assert installed.is_symlink(), f"expected symlink at {installed}"


def test_skills_install_overwrites_existing_target_exit_code(tmp_path: Path):
    # Arrange
    dest = tmp_path / "skills"
    _invoke("install", "--dest", str(dest), "--no-link", "--yes")
    # Act
    result = _invoke("install", "--dest", str(dest), "--yes")
    # Assert
    assert result.exit_code == 0, result.stderr
    installed = dest / "scitex-stats"

def test_skills_install_overwrites_existing_target_is_symlink_installed(tmp_path: Path):
    # Arrange
    dest = tmp_path / "skills"
    _invoke("install", "--dest", str(dest), "--no-link", "--yes")
    # Act
    result = _invoke("install", "--dest", str(dest), "--yes")
    installed = dest / "scitex-stats"
    # Assert
    assert installed.is_symlink()
