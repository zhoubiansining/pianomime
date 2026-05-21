#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" music_lm)"

PYTHON="${MUSIC_LM_PYTHON_BIN:-python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${CONFIG_PYTHON:-python3}"
fi

args=(
  --maestro-root "$MUSIC_LM_MAESTRO_ROOT"
  --out-dir "$MUSIC_LM_TOKENS_DIR"
  --time-step-seconds "$MUSIC_LM_TIME_STEP_SECONDS"
  --max-time-shift-steps "$MUSIC_LM_MAX_TIME_SHIFT_STEPS"
  --velocity-bins "$MUSIC_LM_VELOCITY_BINS"
)

if [[ "${DOWNLOAD:-0}" == "1" ]]; then
  args+=(--download)
fi
if [[ "${LIMIT_PER_SPLIT:-0}" != "0" ]]; then
  args+=(--limit-per-split "$LIMIT_PER_SPLIT")
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
"$PYTHON" -m music_lm.prepare_maestro "${args[@]}"
