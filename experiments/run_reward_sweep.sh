#!/usr/bin/env bash
#
# One-click experiment script for PianoMime reward shaping evaluation.
# Runs single-song policy training across multiple songs and reward variants,
# then produces a comprehensive comparison report.
#
# Usage:
#   # Local (sequential, 1 GPU)
#   bash experiments/run_reward_sweep.sh
#
#   # Cluster (parallel, 8 GPUs, dry run first)
#   CUDA_DEVICES=0,1,2,3,4,5,6,7 bash experiments/run_reward_sweep.sh --dry-run
#   CUDA_DEVICES=0,1,2,3,4,5,6,7 bash experiments/run_reward_sweep.sh
#
#   # Custom songs and variants
#   SONGS="TwinkleTwinkleRousseau Pirates_1" \
#   VARIANTS="baseline velocity_smooth action_smooth" \
#   bash experiments/run_reward_sweep.sh
#

set -euo pipefail

# ---------- defaults ----------
ROOT_DIR=${ROOT_DIR:-/tmp/pianomime_reward_sweep}
CUDA_DEVICES=${CUDA_DEVICES:-0}
TOTAL_ITERS=${TOTAL_ITERS:-300}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
NUM_ENVS=${NUM_ENVS:-8}
N_STEPS=${N_STEPS:-512}
RESIDUAL_FACTOR=${RESIDUAL_FACTOR:-0.03}
SEEDS=${SEEDS:-42}

# Optional JSON config to override / extend reward variant coefficients.
# When set, variants defined in the file are merged on top of built-in defaults
# (existing variants get their coefficients overridden; new variant names are added).
# Example: REWARD_CONFIG=experiments/reward_configs/sensitivity.json
REWARD_CONFIG=${REWARD_CONFIG:-}

# Songs to evaluate (choose 3-5 with available dataset files)
SONGS=${SONGS:-"TwinkleTwinkleRousseau Pirates_1 Stan_1 Petrunko_3"}

# Reward variants to compare
# baseline      : no extra reward shaping (original PianoMime)
# vel_smooth    : penalize joint velocity jerk
# action_smooth : penalize large action changes
# preposition   : reward finger approach before note onset
# timing        : reward precise note onset timing
# finger_dist   : reward finger-key proximity
# smooth_combo  : velocity + action smoothness
# full_combo    : all shaping rewards combined
VARIANTS=${VARIANTS:-"baseline vel_smooth action_smooth preposition timing finger_dist smooth_combo full_combo"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse flags
DRY_RUN=false
SEQUENTIAL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --sequential) SEQUENTIAL=true; shift ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "PianoMime Reward Shaping Experiment"
echo "========================================"
echo "Root dir   : $ROOT_DIR"
echo "CUDA devs  : $CUDA_DEVICES"
echo "Songs      : $SONGS"
echo "Variants   : $VARIANTS"
echo "Seeds      : $SEEDS"
echo "Total iters: $TOTAL_ITERS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Num envs   : $NUM_ENVS"
echo "Residual f : $RESIDUAL_FACTOR"
if [[ -n "$REWARD_CONFIG" ]]; then
    echo "Reward cfg : $REWARD_CONFIG"
fi
echo "========================================"

# Run the sweep
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
    $(if [[ -n "$REWARD_CONFIG" ]]; then echo "--reward-config $REWARD_CONFIG"; fi) \
    $(if $SEQUENTIAL; then echo --sequential; fi) \
    $(if $DRY_RUN; then echo --dry-run; fi)
