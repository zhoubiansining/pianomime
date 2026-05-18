#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TASK_NAME GPU_ID [RUN_ID]" >&2
  exit 2
fi

TASK_NAME="$1"
GPU_ID="$2"
RUN_ID="${3:-manual_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_PROJECT_DIR/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$SCRIPT_PROJECT_DIR/scripts/config_export.py" "$CONFIG_FILE" paths environment)"

RUN_DIR="$RUNTIME_ROOT/runs/multisong_${TASK_NAME}_${RUN_ID}"
HL_LOG="$LOCAL_RESULTS_DIR/multisong/logs/${TASK_NAME}_high_level.log"
LL_LOG="$LOCAL_RESULTS_DIR/multisong/logs/${TASK_NAME}_low_level.log"
VIDEO="$LOCAL_RESULTS_DIR/multisong/videos/${TASK_NAME}_multisong_baseline.mp4"
METRICS="$LOCAL_RESULTS_DIR/multisong/metrics.csv"
TRAJ_DIR="$RUNTIME_DIR/multi_task/trajectories"

log() {
  printf '[%(%F %T)T] [%s:g%s] %s\n' -1 "$TASK_NAME" "$GPU_ID" "$*"
}

parse_metric() {
  local name="$1"
  grep -E "^${name}:" "$LL_LOG" | tail -1 | awk '{print $2}'
}

record_metrics() {
  local precision recall f1 tmp
  precision="$(parse_metric Precision)"
  recall="$(parse_metric Recall)"
  f1="$(parse_metric F1)"
  mkdir -p "$(dirname "$METRICS")"
  if [[ ! -f "$METRICS" ]]; then
    echo "song,split,baseline_stage,precision,recall,f1,video,high_level_log,low_level_log,notes" > "$METRICS"
  fi
  tmp="$(mktemp)"
  grep -v "^${TASK_NAME}," "$METRICS" > "$tmp" || true
  mv "$tmp" "$METRICS"
  echo "${TASK_NAME},test,multisong_checkpoint_eval,${precision},${recall},${f1},multisong/videos/${TASK_NAME}_multisong_baseline.mp4,multisong/logs/${TASK_NAME}_high_level.log,multisong/logs/${TASK_NAME}_low_level.log,manual direct GPU run ${RUN_ID}" >> "$METRICS"
}

mkdir -p "$RUN_DIR" "$TRAJ_DIR" \
  "$LOCAL_RESULTS_DIR/multisong/logs" \
  "$LOCAL_RESULTS_DIR/multisong/videos"

cd "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$RUNTIME_DIR"

log "started"
if [[ -f "$TRAJ_DIR/${TASK_NAME}_left_hand_action_list.npy" && -f "$TRAJ_DIR/${TASK_NAME}_right_hand_action_list.npy" ]]; then
  log "reusing existing high-level trajectories"
  echo "Using existing high-level trajectories for $TASK_NAME" > "$HL_LOG"
else
  log "running high-level diffusion"
  "$PYTHON_BIN" "$RUNTIME_DIR/multi_task/eval_high_level.py" "$TASK_NAME" --config "$CONFIG_FILE" > "$HL_LOG" 2>&1
fi

rm -f 00000.mp4 00001.mp4
log "running low-level diffusion"
"$PYTHON_BIN" "$RUNTIME_DIR/multi_task/eval_low_level.py" "$TASK_NAME" --config "$CONFIG_FILE" > "$LL_LOG" 2>&1

if [[ -f 00001.mp4 ]]; then
  cp 00001.mp4 "$VIDEO"
elif [[ -f 00000.mp4 ]]; then
  cp 00000.mp4 "$VIDEO"
fi

record_metrics

mkdir -p "$RESULTS_DIR/multisong/logs" "$RESULTS_DIR/multisong/videos"
cp -f "$HL_LOG" "$RESULTS_DIR/multisong/logs/" 2>/dev/null || true
cp -f "$LL_LOG" "$RESULTS_DIR/multisong/logs/" 2>/dev/null || true
cp -f "$VIDEO" "$RESULTS_DIR/multisong/videos/" 2>/dev/null || true
cp -f "$METRICS" "$RESULTS_DIR/multisong/metrics.csv" 2>/dev/null || true

log "finished"
