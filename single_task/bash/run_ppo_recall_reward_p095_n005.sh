#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export TOTAL_ITERS="${TOTAL_ITERS:-2000}"
export TAGS="${TAGS:-recall,reward_reweight,weight_sweep}"
export NOTES="${NOTES:-Recall reward weight sweep: positive=0.95, negative=0.05, strict any wrong-key penalty.}"

run_ppo_petrunko "recall_reward_p095_n005" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.95}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.05}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-any}"
