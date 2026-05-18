#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_PROJECT_DIR/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$SCRIPT_PROJECT_DIR/scripts/config_export.py" "$CONFIG_FILE" paths)"

if [[ "$RUNTIME_DIR" == "/" || "$RUNTIME_DIR" == "$PROJECT_DIR" ]]; then
  echo "Refusing unsafe RUNTIME_DIR=$RUNTIME_DIR" >&2
  exit 2
fi

mkdir -p "$RUNTIME_ROOT"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  "$PROJECT_DIR/" "$RUNTIME_DIR/"

printf '%s\n' "$RUNTIME_DIR"
