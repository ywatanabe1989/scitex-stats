"""Pytest fixtures, rootdir marker, and subprocess coverage hook.

An empty conftest.py at tests/ is the canonical SciTeX
convention (audit-project PS208) — it pins the pytest
rootdir and gives downstream fixtures a home.

Subprocess coverage
-------------------
`test_demos.py` and `test_examples_smoke.py` spawn fresh Python
interpreters (`python -m …`, `jupyter nbconvert --execute`). Without
extra wiring those child processes execute uninstrumented, so every
demo line shows up as "missed" even though the test passed.

We enable subprocess tracking the canonical way:

1. `[tool.coverage.run] parallel = true, concurrency = ["multiprocessing"]`
   in `pyproject.toml`. Each subprocess writes its own
   `.coverage.<host>.<pid>` data file.
2. `COVERAGE_PROCESS_START` env var set here so children know which
   config to use.
3. A `.pth` file dropped into the active site-packages with
   `import coverage; coverage.process_startup()`. Python imports
   `.pth` files at interpreter startup, so every child invokes
   `coverage.process_startup()` before user code runs.
4. pytest-cov combines `.coverage.*` files at session end automatically.

The `.pth` install is idempotent and silently no-ops if site-packages
isn't writable (no harm — coverage just stays at the parent-process
number).
"""

from __future__ import annotations

import os
import pathlib
import site

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PTH_NAME = "_scitex_stats_subprocess_coverage.pth"
_PTH_CONTENT = "import coverage; coverage.process_startup()\n"


def _enable_subprocess_coverage() -> None:
    # `setdefault` is a no-op when pytest-cov has already set the env
    # var in its session-start hook (which fires before conftest in
    # some pytest versions). Force-set both so subprocesses inherit
    # the right paths.
    os.environ["COVERAGE_PROCESS_START"] = str(_PROJECT_ROOT / "pyproject.toml")
    # Pin data-file location so children launched with `cwd=tmp_path`
    # still write their `.coverage.<host>.<pid>` next to the parent's
    # `.coverage`, where pytest-cov combines from.
    os.environ["COVERAGE_FILE"] = str(_PROJECT_ROOT / ".coverage")

    candidates: list[pathlib.Path] = []
    for entry in site.getsitepackages():
        candidates.append(pathlib.Path(entry))
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(pathlib.Path(user_site))

    for sp in candidates:
        if not sp.exists() or not os.access(sp, os.W_OK):
            continue
        pth = sp / _PTH_NAME
        if not pth.exists() or pth.read_text() != _PTH_CONTENT:
            try:
                pth.write_text(_PTH_CONTENT)
            except OSError:
                continue
        return


_enable_subprocess_coverage()
