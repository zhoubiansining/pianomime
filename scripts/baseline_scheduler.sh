#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_PROJECT_DIR/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$SCRIPT_PROJECT_DIR/scripts/config_export.py" "$CONFIG_FILE" paths environment scheduler)"

RUN_ID="${RUN_ID:-baseline_$(date +%Y%m%d)}"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_PROJECT_DIR}"

if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ' ')"
fi

LOG_DIR="$LOCAL_RESULTS_DIR/logs"
LOCK_DIR="$LOCAL_RESULTS_DIR/locks"
SCHED_LOG="$LOG_DIR/scheduler_${RUN_ID}.log"

mkdir -p \
  "$LOG_DIR" "$LOCK_DIR" \
  "$LOCAL_RESULTS_DIR/single_song/logs" \
  "$LOCAL_RESULTS_DIR/single_song/videos" \
  "$LOCAL_RESULTS_DIR/single_song/training_runs" \
  "$LOCAL_RESULTS_DIR/multisong/logs" \
  "$LOCAL_RESULTS_DIR/multisong/videos" \
  "$RESULTS_DIR"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$SCHED_LOG"
}

log_err() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$SCHED_LOG" >&2
}

sync_results() {
  rsync -a "$LOCAL_RESULTS_DIR/" "$RESULTS_DIR/"
}

sync_runtime() {
  "$PROJECT_DIR/scripts/setup_artifacts.sh" | tee -a "$SCHED_LOG"
  "$PROJECT_DIR/scripts/sync_to_runtime.sh" | tee -a "$SCHED_LOG"
  mkdir -p "$RUNTIME_DIR/multi_task/trajectories"
}

gpu_mem_used() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

purge_stale_lock() {
  local lock="$1"
  if [[ -f "$lock/pid" ]]; then
    local pid
    pid="$(cat "$lock/pid" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
  fi
  rm -rf "$lock"
  return 0
}

claim_gpu() {
  local label="$1"
  while true; do
    for gpu in $GPU_IDS; do
      local used lock
      used="$(gpu_mem_used "$gpu" || echo 999999)"
      lock="$LOCK_DIR/gpu_${gpu}.lock"
      if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_FREE_MEM_MB )); then
        if mkdir "$lock" 2>/dev/null; then
          printf '%s\n' "$BASHPID" > "$lock/pid"
          log_err "$label claimed GPU $gpu (${used} MiB used)"
          printf '%s\n' "$gpu"
          return 0
        fi
        purge_stale_lock "$lock" || true
      fi
    done
    log_err "$label waiting for GPU; candidates=[$GPU_IDS], threshold=${GPU_FREE_MEM_MB}MiB"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$SCHED_LOG" >&2 || true
    sleep "$POLL_SECONDS"
  done
}

release_gpu() {
  local gpu="$1"
  rm -rf "$LOCK_DIR/gpu_${gpu}.lock"
  log "Released GPU $gpu"
}

metrics_has_song() {
  local metrics="$1"
  local song="$2"
  [[ -f "$metrics" ]] && grep -q "^${song}," "$metrics"
}

append_csv_row() {
  local metrics="$1"
  local song="$2"
  local row="$3"
  local tmp
  tmp="$(mktemp)"
  if [[ -f "$metrics" ]]; then
    grep -v "^${song}," "$metrics" > "$tmp" || true
  fi
  mv "$tmp" "$metrics"
  printf '%s\n' "$row" >> "$metrics"
}

parse_metric() {
  local name="$1"
  local log_file="$2"
  grep -E "^${name}:" "$log_file" | tail -1 | awk '{print $2}'
}

record_single_metrics() {
  local song="$1"
  local log_file="$2"
  local metrics="$LOCAL_RESULTS_DIR/single_song/metrics.csv"
  local precision recall f1
  precision="$(parse_metric Precision "$log_file")"
  recall="$(parse_metric Recall "$log_file")"
  f1="$(parse_metric F1 "$log_file")"
  if [[ ! -f "$metrics" ]]; then
    echo "song,split,baseline_type,precision,recall,f1,video,log,notes" > "$metrics"
  fi
  append_csv_row "$metrics" "$song" \
    "${song},train,single_song_action_replay,${precision},${recall},${f1},single_song/videos/${song}_single_song_baseline.mp4,single_song/logs/${song}.log,action replay with provided trajectory/prior"
}

record_multisong_metrics() {
  local song="$1"
  local hl_log="$2"
  local ll_log="$3"
  local metrics="$LOCAL_RESULTS_DIR/multisong/metrics.csv"
  local precision recall f1
  precision="$(parse_metric Precision "$ll_log")"
  recall="$(parse_metric Recall "$ll_log")"
  f1="$(parse_metric F1 "$ll_log")"
  if [[ ! -f "$metrics" ]]; then
    echo "song,split,baseline_stage,precision,recall,f1,video,high_level_log,low_level_log,notes" > "$metrics"
  fi
  append_csv_row "$metrics" "$song" \
    "${song},test,multisong_checkpoint_eval,${precision},${recall},${f1},multisong/videos/${song}_multisong_baseline.mp4,multisong/logs/${song}_high_level.log,multisong/logs/${song}_low_level.log,checkpoint eval"
}

copy_latest_video() {
  local dest="$1"
  if [[ -f 00001.mp4 ]]; then
    cp 00001.mp4 "$dest"
  elif [[ -f 00000.mp4 ]]; then
    cp 00000.mp4 "$dest"
  fi
}

run_single_replay() {
  local song="$1"
  local metrics="$LOCAL_RESULTS_DIR/single_song/metrics.csv"
  local video="$LOCAL_RESULTS_DIR/single_song/videos/${song}_single_song_baseline.mp4"
  if metrics_has_song "$metrics" "$song" && [[ -f "$video" ]]; then
    log "Skip single-song replay for $song; metrics/video already exist"
    return 0
  fi

  local gpu run_dir log_file
  gpu="$(claim_gpu "single:$song")"
  trap 'release_gpu "$gpu"' EXIT
  run_dir="$RUNTIME_ROOT/runs/single_${song}_${RUN_ID}"
  log_file="$LOCAL_RESULTS_DIR/single_song/logs/${song}.log"
  mkdir -p "$run_dir"

  log "Running single-song replay for $song on GPU $gpu"
  (
    cd "$run_dir"
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl
    export PYTHONPATH="$RUNTIME_DIR:$RUNTIME_DIR/single_task"
    "$PYTHON_BIN" "$RUNTIME_DIR/single_task/test_trained_actions.py" "$song" --config "$CONFIG_FILE" > "$log_file" 2>&1
    copy_latest_video "$video"
  )
  record_single_metrics "$song" "$log_file"
  sync_results
  log "Finished single-song replay for $song"
}

run_multisong_eval() {
  local song="$1"
  local metrics="$LOCAL_RESULTS_DIR/multisong/metrics.csv"
  local video="$LOCAL_RESULTS_DIR/multisong/videos/${song}_multisong_baseline.mp4"
  if metrics_has_song "$metrics" "$song" && [[ -f "$video" ]]; then
    log "Skip multi-song eval for $song; metrics/video already exist"
    return 0
  fi

  local gpu run_dir hl_log ll_log left_traj right_traj
  gpu="$(claim_gpu "multi:$song")"
  trap 'release_gpu "$gpu"' EXIT
  run_dir="$RUNTIME_ROOT/runs/multisong_${song}_${RUN_ID}"
  hl_log="$LOCAL_RESULTS_DIR/multisong/logs/${song}_high_level.log"
  ll_log="$LOCAL_RESULTS_DIR/multisong/logs/${song}_low_level.log"
  left_traj="$RUNTIME_DIR/multi_task/trajectories/${song}_left_hand_action_list.npy"
  right_traj="$RUNTIME_DIR/multi_task/trajectories/${song}_right_hand_action_list.npy"
  mkdir -p "$run_dir" "$RUNTIME_DIR/multi_task/trajectories"

  log "Running multi-song eval for $song on GPU $gpu"
  (
    cd "$run_dir"
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl
    export PYTHONPATH="$RUNTIME_DIR"
    if [[ -f "$left_traj" && -f "$right_traj" ]]; then
      echo "Using existing high-level trajectories for $song" > "$hl_log"
    else
      "$PYTHON_BIN" "$RUNTIME_DIR/multi_task/eval_high_level.py" "$song" --config "$CONFIG_FILE" > "$hl_log" 2>&1
    fi
    rm -f 00000.mp4 00001.mp4
    "$PYTHON_BIN" "$RUNTIME_DIR/multi_task/eval_low_level.py" "$song" --config "$CONFIG_FILE" > "$ll_log" 2>&1
    copy_latest_video "$video"
  )
  rsync -a "$RUNTIME_DIR/multi_task/trajectories/" "$PROJECT_DIR/multi_task/trajectories/" 2>/dev/null || true
  record_multisong_metrics "$song" "$hl_log" "$ll_log"
  sync_results
  log "Finished multi-song eval for $song"
}

run_ppo_training() {
  local song="$1"
  local run_name="${song}_ppo_curve_${RUN_ID}"
  local experiment_dir="$LOCAL_RESULTS_DIR/single_song/training_runs/${run_name}"
  local done_marker="$experiment_dir/.done"
  if [[ -f "$done_marker" && -f "$experiment_dir/eval_metrics.csv" ]]; then
    log "Skip PPO training for $song; done marker already exists"
    return 0
  fi

  local gpu run_dir log_file pretrained maybe_pretrained checkpoint_zip
  gpu="$(claim_gpu "ppo:$song")"
  trap 'release_gpu "$gpu"' EXIT
  run_dir="$RUNTIME_ROOT/runs/ppo_${song}_${RUN_ID}"
  log_file="$LOCAL_RESULTS_DIR/single_song/training_runs/${run_name}.log"
  checkpoint_zip="$run_dir/robopianist_rl/ckpts/${run_name}_best.zip"
  mkdir -p "$run_dir" "$experiment_dir"

  maybe_pretrained=()
  if [[ -f "$checkpoint_zip" ]]; then
    maybe_pretrained=(--pretrained "$checkpoint_zip")
    log "Resuming PPO for $song from $checkpoint_zip"
  fi

  log "Running PPO training for $song on GPU $gpu as $run_name"
  (
    cd "$run_dir"
    export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl
    export PYTHONPATH="$RUNTIME_DIR:$RUNTIME_DIR/single_task"
    export WANDB_MODE=disabled
    "$PYTHON_BIN" "$RUNTIME_DIR/scripts/run_ppo_from_config.py" "$song" \
      --config "$CONFIG_FILE" \
      --root-dir "$LOCAL_RESULTS_DIR/single_song/training_runs" \
      --run-name "$run_name" \
      "${maybe_pretrained[@]}" > "$log_file" 2>&1
  )
  rsync -a "$run_dir/robopianist_rl" "$experiment_dir/" 2>/dev/null || true
  rsync -a "$run_dir/trained_songs" "$experiment_dir/" 2>/dev/null || true
  touch "$done_marker"
  sync_results
  log "Finished PPO training for $song"
}

main() {
  log "Baseline scheduler started"
  log "Shared root: $SHARED_ROOT"
  log "Runtime dir: $RUNTIME_DIR"
  log "Results dir: $RESULTS_DIR"
  log "GPU candidates: $GPU_IDS"
  if [[ "${SKIP_RUNTIME_SYNC:-0}" == "1" ]]; then
    log "Skipping runtime sync because SKIP_RUNTIME_SYNC=1"
  else
    sync_runtime
  fi
  sync_results

  local pids=()
  for song in $SINGLE_REPLAY_TASKS; do
    ( run_single_replay "$song" ) &
    pids+=("$!")
  done
  for song in $MULTISONG_TASKS; do
    ( run_multisong_eval "$song" ) &
    pids+=("$!")
  done
  for song in $PPO_TASKS; do
    ( run_ppo_training "$song" ) &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  sync_results
  log "Baseline scheduler finished with status $status"
  return "$status"
}

main "$@"
