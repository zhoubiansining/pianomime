#!/usr/bin/env bash
#
# Sensitivity analysis for individual reward shaping coefficients.
#
# For each reward term, runs baseline + that term at several coefficient
# values on a single song.  The goal is to find the best coefficient per
# term before combining them in a full sweep.
#
# Usage:
#   bash experiments/run_sensitivity.sh
#
#   # Custom song / faster run
#   SENSITIVITY_SONG=Stan_1 TOTAL_ITERS=150 bash experiments/run_sensitivity.sh
#
#   # With custom coefficient ranges (edit reward_configs/sensitivity.json first)
#   REWARD_CONFIG=experiments/reward_configs/sensitivity.json bash experiments/run_sensitivity.sh
#
set -euo pipefail

# ---------- configurable defaults ----------
ROOT_DIR=${ROOT_DIR:-/tmp/pianomime_sensitivity}
CUDA_DEVICES=${CUDA_DEVICES:-0}
TOTAL_ITERS=${TOTAL_ITERS:-200}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
NUM_ENVS=${NUM_ENVS:-8}
N_STEPS=${N_STEPS:-512}
RESIDUAL_FACTOR=${RESIDUAL_FACTOR:-0.03}
SEEDS=${SEEDS:-42}

# Single song for sensitivity (pick a short one to save time)
SENSITIVITY_SONG=${SENSITIVITY_SONG:-TwinkleTwinkleRousseau}

# Reward config (built-in sensitivity ranges)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REWARD_CONFIG=${REWARD_CONFIG:-$SCRIPT_DIR/reward_configs/sensitivity.json}

# Coefficient groups: "<group_label> <variant1 variant2 ...>"
# Each group is one reward term swept across coefficient values.
# Edit reward_configs/sensitivity.json to change the actual coefficients.
COEFF_GROUPS=(
  "timing       timing_0.1 timing_0.5 timing_1.0 timing_2.0"
  "preposition  prep_0.1 prep_0.3 prep_0.5 prep_1.0"
  "vel_smooth   vel_sm_0.001 vel_sm_0.01 vel_sm_0.1"
  "act_smooth   act_sm_0.001 act_sm_0.01 act_sm_0.1"
  "finger_dist  fdist_0.1 fdist_0.5 fdist_1.0"
)

# ---------- helpers ----------
run_group() {
  local label="$1"; shift
  local variants=("$@")

  echo ""
  echo "========================================"
  echo "  Group: $label"
  echo "  Variants: ${variants[*]}"
  echo "  Song: $SENSITIVITY_SONG"
  echo "========================================"

  cd "$PROJECT_ROOT"

  python "$SCRIPT_DIR/reward_shaping_sweep.py" \
    --root-dir "$ROOT_DIR" \
    --cuda-devices ${CUDA_DEVICES//,/ } \
    --total-iters "$TOTAL_ITERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --num-envs "$NUM_ENVS" \
    --n-steps "$N_STEPS" \
    --residual-factor "$RESIDUAL_FACTOR" \
    --songs "$SENSITIVITY_SONG" \
    --variants baseline "${variants[@]}" \
    --seeds ${SEEDS} \
    --reward-config "$REWARD_CONFIG"
}

# ---------- main ----------
echo "========================================"
echo "Reward Coefficient Sensitivity Analysis"
echo "========================================"
echo "Root dir   : $ROOT_DIR"
echo "CUDA devs  : $CUDA_DEVICES"
echo "Song       : $SENSITIVITY_SONG"
echo "Total iters: $TOTAL_ITERS"
echo "Eval interv: $EVAL_INTERVAL"
echo "Seeds      : $SEEDS"
echo "Reward cfg : $REWARD_CONFIG"
echo "Groups     : ${#COEFF_GROUPS[@]}"
echo "========================================"

if [[ ! -f "$REWARD_CONFIG" ]]; then
  echo "[ERROR] Reward config not found: $REWARD_CONFIG"
  exit 1
fi

START_TIME=$(date +%s)

for group_def in "${COEFF_GROUPS[@]}"; do
  # Split into label + variants: "timing timing_0.1 timing_0.5 ..."
  read -r label variants_str <<< "$group_def"
  read -ra variants <<< "$variants_str"
  run_group "$label" "${variants[@]}"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "========================================"
echo "All sensitivity groups finished."
echo "Elapsed: ${MINUTES}m ${SECONDS}s"
echo "Results in: $ROOT_DIR"
echo "========================================"

# ---------- quick per-group summary ----------
echo ""
echo "=== Per-Term Best Coefficient ==="
echo ""

cd "$PROJECT_ROOT"

for group_def in "${COEFF_GROUPS[@]}"; do
  read -r label variants_str <<< "$group_def"
  read -ra variants <<< "$variants_str"

  best_variant=""
  best_f1=-999

  for v in "${variants[@]}"; do
    run_dir="$ROOT_DIR/${v}__${SENSITIVITY_SONG}"
    result_json="$run_dir/result.json"
    if [[ -f "$result_json" ]]; then
      f1=$(python3 -c "
import json
d = json.load(open('$result_json'))
m = d.get('metrics', {})
print(m.get('f1', -1))
" 2>/dev/null || echo "-1")
      if [[ "$f1" != "-1" ]]; then
        # Use awk for float comparison
        better=$(awk -v a="$f1" -v b="$best_f1" 'BEGIN { print (a > b) ? "1" : "0" }')
        if [[ "$better" == "1" ]]; then
          best_f1="$f1"
          best_variant="$v"
        fi
      fi
    fi
  done

  baseline_dir="$ROOT_DIR/baseline__${SENSITIVITY_SONG}"
  baseline_f1="n/a"
  if [[ -f "$baseline_dir/result.json" ]]; then
    baseline_f1=$(python3 -c "
import json
d = json.load(open('$baseline_dir/result.json'))
m = d.get('metrics', {})
print(round(m.get('f1', 0), 4))
" 2>/dev/null || echo "n/a")
  fi

  printf "%-14s  baseline F1: %8s  |  best: %-16s  F1: %s\n" \
    "$label" "$baseline_f1" "$best_variant" "$best_f1"
done

echo ""
echo "Use the best coefficient for each term in a full-combo sweep."
