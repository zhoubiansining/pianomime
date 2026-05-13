#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT="${SHARED_ROOT:-/home/gaoj/share4/_piano}"
PROJECT_DIR="${PROJECT_DIR:-$SHARED_ROOT/pianomime}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/gaoj/piano_scratch}"
RUNTIME_DIR="${RUNTIME_DIR:-$RUNTIME_ROOT/pianomime}"

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
