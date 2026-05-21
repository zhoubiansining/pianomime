#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SONG_NAME [RUN_NAME]" >&2
  exit 2
fi

SONG="$1"
RUN_NAME="${2:-${SONG}_ppo_music_lm_$(date +%Y%m%d_%H%M%S)}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" paths environment music_lm)"

PYTHON="${PYTHON_BIN:-python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${CONFIG_PYTHON:-python3}"
fi

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$MUSIC_LM_CHECKPOINT" ]]; then
  echo "Missing music LM checkpoint: $MUSIC_LM_CHECKPOINT" >&2
  echo "Run scripts/train_music_lm.sh first, or set MUSIC_LM_CHECKPOINT=/path/to/best.pt." >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/single_task:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES}"

args=(
  "$PROJECT_ROOT/scripts/run_ppo_from_config.py" "$SONG"
  --config "$CONFIG_FILE" \
  --run-name "$RUN_NAME" \
  --music-lm-checkpoint "$MUSIC_LM_CHECKPOINT" \
  --music-lm-reward-weight "$MUSIC_LM_PPO_REWARD_WEIGHT" \
  --music-lm-reward-window-tokens "$MUSIC_LM_PPO_REWARD_WINDOW_TOKENS" \
  --music-lm-reward-clip "$MUSIC_LM_PPO_REWARD_CLIP"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON" "${args[@]}"
