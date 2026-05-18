#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" paths environment)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/single_task:${PYTHONPATH:-}"

SONG="${1:-}"
if [[ -z "$SONG" ]]; then
  SONG="$(CONFIG_FILE="$CONFIG_FILE" "${CONFIG_PYTHON:-python3}" - <<'PY'
from pianomime_config import load_config, section
cfg = load_config()
songs = section(cfg, "single_song").get("ppo_songs", [])
if not songs:
    raise SystemExit("single_song.ppo_songs is empty")
print(songs[0])
PY
)"
fi

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPU_ID}"

cd "${RUN_DIR:-$PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE_ID"

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_ppo_from_config.py" "$SONG" --config "$CONFIG_FILE" "${extra_args[@]}"
