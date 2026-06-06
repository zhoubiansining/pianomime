#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

run_ppo_petrunko "cosine_lr" \
    --lr-schedule-type "${LR_SCHEDULE_TYPE:-cosine}" \
    --final-lr "${FINAL_LR:-2e-5}" \
    --lr-warmup-iters "${LR_WARMUP_ITERS:-50}" \
    --lr-warmup-start-factor "${LR_WARMUP_START_FACTOR:-0.25}"
