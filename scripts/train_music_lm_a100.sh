#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" music_lm)"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export DEVICE="${DEVICE:-cuda}"
export MAX_STEPS="${MAX_STEPS:-$MUSIC_LM_MAX_STEPS}"
export A100_AUTO_SETUP="${A100_AUTO_SETUP:-1}"
export A100_DOWNLOAD_MAESTRO="${A100_DOWNLOAD_MAESTRO:-0}"
export A100_LIMIT_PER_SPLIT="${A100_LIMIT_PER_SPLIT:-0}"

log() {
  printf '[music-lm-a100] %s\n' "$*"
}

if [[ "$A100_AUTO_SETUP" == "1" && ! -x "$MUSIC_LM_PYTHON_BIN" ]]; then
  log "Creating Music LM virtualenv"
  bash "$PROJECT_ROOT/scripts/setup_music_lm_env.sh"
fi

PYTHON="${MUSIC_LM_PYTHON_BIN:-python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${CONFIG_PYTHON:-python3}"
fi

"$PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Run this script on an A100/CUDA node or set DEVICE=cpu for debugging.")

name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"cuda_device={name}")
print(f"cuda_capability={capability[0]}.{capability[1]}")
if "A100" not in name:
    print(f"warning: expected an A100, found {name}", file=sys.stderr)
PY

if [[ ! -f "$MUSIC_LM_TOKENS_DIR/train.bin" ]]; then
  if [[ "$A100_DOWNLOAD_MAESTRO" != "1" ]]; then
    cat >&2 <<EOF
Missing tokenized MAESTRO data at:
  $MUSIC_LM_TOKENS_DIR/train.bin

Either copy artifacts/maestro_tokens to this machine, or rerun with:
  A100_DOWNLOAD_MAESTRO=1 bash scripts/train_music_lm_a100.sh
EOF
    exit 2
  fi

  log "Preparing MAESTRO tokens on this node"
  DOWNLOAD=1 LIMIT_PER_SPLIT="$A100_LIMIT_PER_SPLIT" bash "$PROJECT_ROOT/scripts/prepare_music_lm_data.sh"
fi

log "Starting Music LM training on $DEVICE"
log "Output: $MUSIC_LM_OUTPUT_DIR"
log "Steps: $MAX_STEPS"

bash "$PROJECT_ROOT/scripts/train_music_lm.sh"
