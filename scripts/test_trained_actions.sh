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
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" paths environment)"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPU_ID}"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
MUJOCO_GL="${MUJOCO_GL:-egl}" \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE_ID" \
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" "$PROJECT_ROOT/single_task/test_trained_actions.py" "$ARGUMENT" --config "$CONFIG_FILE"
