#!/bin/bash
# File: examples/00_run_all.sh
# Run all examples

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h]

Run all SciTeX Stats example scripts.

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

for script in "$SCRIPT_DIR"/[0-9][0-9]_*.py; do
    [ -f "$script" ] || continue
    echo "--- Running: $(basename "$script") ---"
    python "$script"
    echo
done

echo "=== All examples completed ==="
