#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT}"
ROOT_DIR="${ROOT_DIR:-/tmp/robopianist/rl/}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPU_ID}"

cd "$RUN_DIR"

WANDB_DIR="${WANDB_DIR:-/tmp/robopianist/}" \
MUJOCO_GL="${MUJOCO_GL:-egl}" \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE_ID" \
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" "$PROJECT_ROOT/single_task/train_ppo.py" \
    --root-dir "$ROOT_DIR" \
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
    --num-envs 32 \
    --initial-lr 3e-4 \
    --lr-decay-rate 0.999 \
    --n-steps 512 \
    --mimic-task "Petrunko_3" \
    --environment-name "Petrunko_3" \
    --use-note-trajectory \
    --total-iters 2000 \
    --residual-factor 0.03 \
    --deepmimic \
    
