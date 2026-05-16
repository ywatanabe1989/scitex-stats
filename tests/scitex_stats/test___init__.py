#!/usr/bin/env python3
# File: tests/scitex_stats/test___init__.py

"""Sanity tests for the .env-respect + runtime-separation pattern.

Each test covers exactly one observable contract of the path-resolver
convention introduced in ``src/scitex_stats/__init__.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_exposes_runtime_path_callable():
    # Arrange
    import scitex_stats

    # Act
    fn = getattr(scitex_stats, "_runtime_path", None)

    # Assert
    assert callable(fn)


def test_runtime_path_returns_pathlib_path():
    # Arrange
    from scitex_stats import _runtime_path

    # Act
    p = _runtime_path("cache", "demo.db")

    # Assert
    assert isinstance(p, Path)


def test_runtime_path_lives_under_scitex_stats_runtime_dir():
    # Arrange
    from scitex_stats import _runtime_path

    expected_prefix = Path.home() / ".scitex" / "stats" / "runtime"

    # Act
    p = _runtime_path("cache", "demo.db")

    # Assert
    assert str(p).startswith(str(expected_prefix))


def test_runtime_path_preserves_trailing_filename():
    # Arrange
    from scitex_stats import _runtime_path

    # Act
    p = _runtime_path("cache", "demo.db")

    # Assert
    assert p.name == "demo.db"


def test_runtime_path_preserves_intermediate_subdir():
    # Arrange
    from scitex_stats import _runtime_path

    # Act
    p = _runtime_path("cache", "demo.db")

    # Assert
    assert p.parent.name == "cache"


def test_migrate_runtime_dirs_moves_legacy_subdir_to_runtime(tmp_path):
    # Arrange: run the migration in a subprocess with HOME=tmp_path so the
    # real ``~/.scitex/stats/`` is never touched. Real env var, real
    # filesystem state — no monkeypatching of production internals.
    legacy = tmp_path / ".scitex" / "stats" / "cache"
    legacy.mkdir(parents=True)
    (legacy / "marker.txt").write_text("legacy")
    env = {**os.environ, "HOME": str(tmp_path)}

    # Act
    subprocess.run(
        [sys.executable, "-c", "import scitex_stats"],
        env=env,
        check=True,
    )

    # Assert
    moved = tmp_path / ".scitex" / "stats" / "runtime" / "cache" / "marker.txt"
    assert moved.read_text() == "legacy"


def test_migrate_runtime_dirs_is_idempotent(tmp_path):
    # Arrange: HOME with the post-migration layout already in place;
    # a second import must leave the runtime/ tree untouched.
    runtime_cache = tmp_path / ".scitex" / "stats" / "runtime" / "cache"
    runtime_cache.mkdir(parents=True)
    (runtime_cache / "marker.txt").write_text("already-migrated")
    env = {**os.environ, "HOME": str(tmp_path)}

    # Act
    subprocess.run(
        [sys.executable, "-c", "import scitex_stats; import scitex_stats"],
        env=env,
        check=True,
    )

    # Assert
    assert (runtime_cache / "marker.txt").read_text() == "already-migrated"
