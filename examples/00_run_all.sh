#!/bin/bash
# File: examples/00_run_all.sh
# Re-execute every example notebook in place (refreshes outputs).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h]

Re-execute every example notebook in place via jupytext.

Options:
  -h, --help    Show this help message and exit
EOF
}

case "${1:-}" in
-h | --help)
    usage
    exit 0
    ;;
esac

echo "=== SciTeX Stats Examples ==="
echo

for nb in "$SCRIPT_DIR"/[0-9][0-9]_*.ipynb; do
    [ -f "$nb" ] || continue
    echo "--- Executing: $(basename "$nb") ---"
    jupytext --execute --to ipynb "$nb" -o "$nb"
    echo
done

echo "=== All notebooks executed ==="
