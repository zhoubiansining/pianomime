#!/usr/bin/env bash
set -euo pipefail

VENV=/home/gaoj/share4/_piano/.venv
REPO=/home/gaoj/piano_scratch/pianomime
LOCAL_RESULTS=/home/gaoj/piano_scratch/baseline_results
SHARE_RESULTS=/home/gaoj/share4/_piano/baseline_results
STAMP=${PIANOMIME_STAMP:-$(date +%Y%m%d_%H%M%S)}
SCHED_LOG="$LOCAL_RESULTS/gpu_scheduler_${STAMP}.log"

mkdir -p "$LOCAL_RESULTS/multisong/logs" "$LOCAL_RESULTS/multisong/videos"
mkdir -p "$LOCAL_RESULTS/single_song/training_runs"
mkdir -p "$SHARE_RESULTS/multisong/logs" "$SHARE_RESULTS/multisong/videos"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$SCHED_LOG"
}

sync_results() {
  rsync -a "$LOCAL_RESULTS/" "$SHARE_RESULTS/"
}

gpu_mem_used() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

wait_for_gpu() {
  local gpu="$1"
  local max_mem="${2:-5000}"
  while true; do
    local used
    used=$(gpu_mem_used "$gpu" || echo 999999)
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < max_mem )); then
      log "GPU $gpu is free enough: ${used}MiB < ${max_mem}MiB"
      return 0
    fi
    log "GPU $gpu busy: ${used}MiB used; waiting"
    sleep 60
  done
}

append_multisong_metrics() {
  local song="$1"
  local video="$2"
  local hl_log="$3"
  local ll_log="$4"
  local metrics="$LOCAL_RESULTS/multisong/metrics.csv"
  local precision recall f1 tmp

  precision=$(grep -E "^Precision:" "$ll_log" | tail -1 | awk '{print $2}')
  recall=$(grep -E "^Recall:" "$ll_log" | tail -1 | awk '{print $2}')
  f1=$(grep -E "^F1:" "$ll_log" | tail -1 | awk '{print $2}')

  if [[ ! -f "$metrics" ]]; then
    echo "song,split,baseline_stage,precision,recall,f1,video,high_level_log,low_level_log,notes" > "$metrics"
  fi

  tmp=$(mktemp)
  grep -v "^${song}," "$metrics" > "$tmp" || true
  mv "$tmp" "$metrics"
  echo "${song},test,multisong_checkpoint_eval,${precision},${recall},${f1},multisong/videos/${song}_multisong_baseline.mp4,multisong/logs/${song}_high_level.log,multisong/logs/${song}_low_level.log,checkpoint eval on GPU" >> "$metrics"
}

run_multisong_eval() {
  local gpu="$1"
  local song="$2"
  local run_dir=/home/gaoj/piano_scratch/runs/multisong_${song}_${STAMP}
  local hl_log="$LOCAL_RESULTS/multisong/logs/${song}_high_level.log"
  local ll_log="$LOCAL_RESULTS/multisong/logs/${song}_low_level.log"
  local video="$LOCAL_RESULTS/multisong/videos/${song}_multisong_baseline.mp4"

  wait_for_gpu "$gpu"
  mkdir -p "$run_dir"
  log "Starting multi-song high-level eval for $song on GPU $gpu"
  (
    cd "$run_dir"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MUJOCO_EGL_DEVICE_ID="$gpu"
    export MUJOCO_GL=egl
    export PYTHONPATH="$REPO"
    "$VENV/bin/python" "$REPO/multi_task/eval_high_level.py" "$song" > "$hl_log" 2>&1
    rm -f 00000.mp4 00001.mp4
    log "Starting multi-song low-level eval for $song on GPU $gpu"
    "$VENV/bin/python" "$REPO/multi_task/eval_low_level.py" "$song" > "$ll_log" 2>&1
    if [[ -f 00001.mp4 ]]; then
      cp 00001.mp4 "$video"
    elif [[ -f 00000.mp4 ]]; then
      cp 00000.mp4 "$video"
    fi
    append_multisong_metrics "$song" "$video" "$hl_log" "$ll_log"
    sync_results
    log "Finished multi-song eval for $song"
  ) >> "$SCHED_LOG" 2>&1
}

run_ppo_training() {
  local gpu="$1"
  local song="$2"
  local run_name="${song}_ppo_curve_${STAMP}"
  local run_dir=/home/gaoj/piano_scratch/runs/ppo_${song}_${STAMP}
  local log_file="$LOCAL_RESULTS/single_song/training_runs/${run_name}.log"
  local artifacts="$LOCAL_RESULTS/single_song/training_runs/${run_name}_artifacts"

  wait_for_gpu "$gpu"
  mkdir -p "$run_dir" "$artifacts"
  log "Starting PPO training for $song on GPU $gpu as $run_name"
  (
    cd "$run_dir"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export MUJOCO_EGL_DEVICE_ID="$gpu"
    export MUJOCO_GL=egl
    export PYTHONPATH="$REPO:$REPO/single_task"
    export WANDB_MODE=disabled
    "$VENV/bin/python" "$REPO/single_task/train_ppo.py" \
      --root-dir "$LOCAL_RESULTS/single_song/training_runs" \
      --name "$run_name" \
      --warmstart-steps 5000 \
      --max-steps 1000000 \
      --discount 0.99 \
      --trim-silence \
      --gravity-compensation \
      --control-timestep 0.05 \
      --n-steps-lookahead 0 \
      --disable_fingering_reward \
      --disable_hand_collisions \
      --disable_forearm_reward \
      --tqdm-bar \
      --eval-episodes 1 \
      --camera-id "piano/back" \
      --midi-start-from 0 \
      --residual-action \
      --frame-stack 4 \
      --num-envs 1 \
      --initial-lr 3e-4 \
      --lr-decay-rate 0.999 \
      --n-steps 512 \
      --mimic-task "$song" \
      --environment-name "$song" \
      --use-note-trajectory \
      --total-iters 2000 \
      --residual-factor 0.03 \
      --deepmimic > "$log_file" 2>&1
    cp -a "$run_dir/robopianist_rl" "$artifacts/" 2>/dev/null || true
    cp -a "$run_dir/trained_songs" "$artifacts/" 2>/dev/null || true
    sync_results
    log "Finished PPO training for $song"
  ) >> "$SCHED_LOG" 2>&1
}

log "Scheduler started; results will sync to $SHARE_RESULTS"
run_multisong_eval 5 Numb_1 &
run_multisong_eval 6 NoTimeToDie_1 &
run_ppo_training 7 Petrunko_3 &
wait
sync_results
log "All scheduled jobs finished"
