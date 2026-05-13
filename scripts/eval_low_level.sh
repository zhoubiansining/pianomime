#!/bin/bash
set -euo pipefail

# Check if the song name is given
if [ -z "$1" ]; then
  echo "No song name given"
  exit 1
fi

# Capture the argument
ARGUMENT=$1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPU_ID}"

WANDB_DIR="${WANDB_DIR:-/tmp/robopianist/}" \
MUJOCO_GL="${MUJOCO_GL:-egl}" \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE_ID" \
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" "$PROJECT_ROOT/multi_task/eval_low_level.py" "$ARGUMENT"
