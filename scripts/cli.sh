#!/usr/bin/env bash
# cli.sh — Run EvolveBot locally (no Docker needed)
#
# Usage:
#   scripts/cli.sh                    Interactive mode
#   scripts/cli.sh "write a script"   Pass goal directly
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "[cli] Creating venv..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -r "$PROJECT_DIR/requirements.txt"
fi

exec "$VENV/bin/python" "$PROJECT_DIR/src/main.py" "$@"
