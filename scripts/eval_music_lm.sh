#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MIDI_FILE [MIDI_FILE ...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" music_lm)"

PYTHON="${MUSIC_LM_PYTHON_BIN:-python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${CONFIG_PYTHON:-python3}"
fi

CHECKPOINT="${CHECKPOINT:-$MUSIC_LM_CHECKPOINT}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing music LM checkpoint: $CHECKPOINT" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
"$PYTHON" -m music_lm.evaluate --checkpoint "$CHECKPOINT" "$@"
