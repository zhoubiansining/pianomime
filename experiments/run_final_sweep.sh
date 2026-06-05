#!/usr/bin/env bash
#
# Final sweep: runs preset reward variants across multiple songs.
#
# The 8 variants are:
#   1. baseline
#   2. baseline + best_timing          (timing coef=1.0, ΔF1=+0.0227)
#   3. baseline + best_preposition      (prep coef=1.0,   ΔF1=+0.0152)
#   4. baseline + best_vel_smooth       (vel_sm coef=0.001, ΔF1=+0.0085)
#   5. baseline + best_finger_dist     (fdist coef=1.0,  ΔF1=+0.0133)
#   6. baseline + best_top3_combo      (timing+prep+fdist, all coef=1.0)
#   7. baseline + best_full_combo      (all 4 improved terms)
#   8. baseline + best_act_smooth      (act_smooth coef=0.001, ΔF1=-0.0185, NOT improved)
#      Note: best_act_smooth is included for completeness but expected to hurt.
#
# Usage:
#   bash experiments/run_final_sweep.sh
#
#   # Override some settings
#   ROOT_DIR=/data/final_sweep \
#   CUDA_DEVICES=0,1,2,3 \
#   TOTAL_ITERS=300 \
#   bash experiments/run_final_sweep.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- defaults ----------
ROOT_DIR=${ROOT_DIR:-/tmp/pianomime_final_sweep}
CUDA_DEVICES=${CUDA_DEVICES:-0}
TOTAL_ITERS=${TOTAL_ITERS:-300}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
NUM_ENVS=${NUM_ENVS:-8}
N_STEPS=${N_STEPS:-512}
RESIDUAL_FACTOR=${RESIDUAL_FACTOR:-0.03}
SEEDS=${SEEDS:-42}

SONGS=${SONGS:-"TwinkleTwinkleRousseau Pirates_1 Stan_1 Petrunko_3"}

# Path to best-coefficient config (pre-filled from sensitivity analysis).
REWARD_CONFIG=${REWARD_CONFIG:-$SCRIPT_DIR/reward_configs/best.json}

# Which combo variants to run. All exclude act_smooth since sensitivity
# showed all its coefficients hurt performance.
VARIANTS=${VARIANTS:-"baseline best_timing best_preposition best_vel_smooth best_finger_dist best_top3_combo best_full_combo"}

# ---------- main ----------
echo "========================================"
echo "Final Reward Shaping Sweep"
echo "========================================"
echo "Root dir   : $ROOT_DIR"
echo "CUDA devs  : $CUDA_DEVICES"
echo "Songs      : $SONGS"
echo "Variants   : $VARIANTS"
echo "Seeds      : $SEEDS"
echo "Total iters: $TOTAL_ITERS"
echo "Eval interv: $EVAL_INTERVAL"
echo "Residual f : $RESIDUAL_FACTOR"
echo "Reward cfg : $REWARD_CONFIG"
echo "========================================"

if [[ ! -f "$REWARD_CONFIG" ]]; then
    echo "[ERROR] Reward config not found: $REWARD_CONFIG"
    exit 1
fi

cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/reward_shaping_sweep.py" \
    --root-dir "$ROOT_DIR" \
    --cuda-devices ${CUDA_DEVICES//,/ } \
    --total-iters "$TOTAL_ITERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --num-envs "$NUM_ENVS" \
    --n-steps "$N_STEPS" \
    --residual-factor "$RESIDUAL_FACTOR" \
    --songs ${SONGS} \
    --variants ${VARIANTS} \
    --seeds ${SEEDS} \
    --reward-config "$REWARD_CONFIG" \
    "$@"
