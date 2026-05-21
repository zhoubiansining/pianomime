#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" music_lm)"

PYTHON="${MUSIC_LM_PYTHON_BIN:-python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${CONFIG_PYTHON:-python3}"
fi

if [[ ! -f "$MUSIC_LM_TOKENS_DIR/train.bin" ]]; then
  echo "Missing tokenized MAESTRO data at $MUSIC_LM_TOKENS_DIR/train.bin" >&2
  echo "Run scripts/prepare_music_lm_data.sh first." >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
args=(
  --data-dir "$MUSIC_LM_TOKENS_DIR"
  --out-dir "$MUSIC_LM_OUTPUT_DIR"
  --block-size "$MUSIC_LM_BLOCK_SIZE"
  --batch-size "$MUSIC_LM_BATCH_SIZE"
  --n-layer "$MUSIC_LM_N_LAYER"
  --n-head "$MUSIC_LM_N_HEAD"
  --n-embd "$MUSIC_LM_N_EMBD"
  --dropout "$MUSIC_LM_DROPOUT"
  --learning-rate "$MUSIC_LM_LEARNING_RATE"
  --weight-decay "$MUSIC_LM_WEIGHT_DECAY"
  --max-steps "${MAX_STEPS:-$MUSIC_LM_MAX_STEPS}"
  --eval-interval "$MUSIC_LM_EVAL_INTERVAL"
  --eval-iters "$MUSIC_LM_EVAL_ITERS"
  --grad-clip "$MUSIC_LM_GRAD_CLIP"
  --seed "$MUSIC_LM_SEED"
)
if [[ -n "${DEVICE:-}" ]]; then
  args+=(--device "$DEVICE")
fi

"$PYTHON" -m music_lm.train "${args[@]}"
