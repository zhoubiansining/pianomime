#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export TOTAL_ITERS="${TOTAL_ITERS:-2000}"
export TAGS="${TAGS:-recall,large_policy,reward_reweight}"
export NOTES="${NOTES:-Combine large policy capacity with the best recall_reward reweighting while keeping strict any wrong-key penalty.}"
export POLICY_PI_ARCH="${POLICY_PI_ARCH:-2048,1024,512}"
export POLICY_VF_ARCH="${POLICY_VF_ARCH:-2048,1024,512}"

run_ppo_petrunko "large_policy_recall_reward" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.8}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.2}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-any}"
