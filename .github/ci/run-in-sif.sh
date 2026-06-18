#!/usr/bin/env bash
# Runs INSIDE the reused CI SIF (apptainer exec). $1 = python version.
#
# The SIF has the ecosystem dependency set installed and is READ-ONLY. CI must
# run the CHECKOUT's code, so we prepend it on PYTHONPATH — that shadows any
# baked copy for imports + coverage.
#
# No install, no --writable-tmpfs: nothing is written into the SIF (the baked
# venv is root-owned — a runtime install hits Permission denied even on a
# tmpfs overlay).
#
# Fail-loud: a SIF without the venv baked is a hard error (rebuild the SIF).
set -euo pipefail

V="${1:?python version arg required (3.11/3.12/3.13)}"
VENV="/opt/venv-$V"
test -x "$VENV/bin/python" || {
    echo "::error::baked venv python missing in $VENV — rebuild: scitex-container apptainer build ci-cpu"
    exit 1
}

export LC_ALL=C.UTF-8 LANG=C.UTF-8

# Real writable scratch. The runner profile exports TMPDIR=~/.cache/tmp, a host
# path that does NOT resolve inside the container; tests (tmp_path) and mktemp
# need a working tmp. Node-local /tmp is writable + ephemeral.
export TMPDIR="/tmp/ci-$V"
mkdir -p "$TMPDIR"

# A VIRTUAL_ENV leaked from the runner profile (~/.env-3.11) is a broken symlink
# in here; unset it so no tool (uv, pip) tries to follow it.
unset VIRTUAL_ENV || true

# venv bin on PATH (python3, pytest); PYTHONPATH prepends the checkout so
# imports + coverage use the PR code.
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PWD/src"

echo "py=$("$VENV"/bin/python -V) pytest=$(command -v pytest)"
exec pytest tests/ --cov=src/scitex_stats --cov-report=xml --cov-report=term
