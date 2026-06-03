#!/usr/bin/env bash
set -euo pipefail

SONG="${1:-Petrunko_3}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-music_lm_ablation_${SONG}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
CONFIG_PYTHON="${CONFIG_PYTHON:-python3}"
eval "$("$CONFIG_PYTHON" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" paths music_lm)"

PYTHON="${PYTHON_BIN:-python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${CONFIG_PYTHON:-python3}"
fi

RESULT_ROOT="${RESULT_ROOT:-$LOCAL_RESULTS_DIR/single_song/training_runs}"
NO_LM_RUN="${NO_LM_RUN:-${RUN_ID}_no_music_lm}"
WITH_LM_RUN="${WITH_LM_RUN:-${RUN_ID}_with_music_lm}"

COMMON_ARGS=(
  "$PROJECT_ROOT/scripts/run_ppo_from_config.py" "$SONG"
  --config "$CONFIG_FILE"
  --root-dir "$RESULT_ROOT"
)

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/single_task:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "$PYTHON" "${COMMON_ARGS[@]}" --run-name "$NO_LM_RUN" --dry-run
  "$PYTHON" "${COMMON_ARGS[@]}" --run-name "$WITH_LM_RUN" \
    --music-lm-checkpoint "$MUSIC_LM_CHECKPOINT" \
    --music-lm-reward-weight "${MUSIC_LM_REWARD_WEIGHT:-$MUSIC_LM_PPO_REWARD_WEIGHT}" \
    --music-lm-reward-window-tokens "$MUSIC_LM_PPO_REWARD_WINDOW_TOKENS" \
    --music-lm-reward-clip "$MUSIC_LM_PPO_REWARD_CLIP" \
    --dry-run
  exit 0
fi

if [[ ! -f "$MUSIC_LM_CHECKPOINT" ]]; then
  echo "Missing music LM checkpoint: $MUSIC_LM_CHECKPOINT" >&2
  exit 2
fi

printf '[ablation] no-LM run: %s\n' "$NO_LM_RUN"
"$PYTHON" "${COMMON_ARGS[@]}" --run-name "$NO_LM_RUN"

printf '[ablation] with-LM run: %s\n' "$WITH_LM_RUN"
"$PYTHON" "${COMMON_ARGS[@]}" --run-name "$WITH_LM_RUN" \
  --music-lm-checkpoint "$MUSIC_LM_CHECKPOINT" \
  --music-lm-reward-weight "${MUSIC_LM_REWARD_WEIGHT:-$MUSIC_LM_PPO_REWARD_WEIGHT}" \
  --music-lm-reward-window-tokens "$MUSIC_LM_PPO_REWARD_WINDOW_TOKENS" \
  --music-lm-reward-clip "$MUSIC_LM_PPO_REWARD_CLIP"

"$PYTHON" "$PROJECT_ROOT/scripts/summarize_music_lm_ablation.py" \
  "$RESULT_ROOT/$NO_LM_RUN/eval_metrics.csv" \
  "$RESULT_ROOT/$WITH_LM_RUN/eval_metrics.csv"
