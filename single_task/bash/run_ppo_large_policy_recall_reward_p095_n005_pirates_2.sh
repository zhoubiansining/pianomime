#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export TOTAL_ITERS="${TOTAL_ITERS:-2000}"
export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Pirates_2}"
export TAGS="${TAGS:-recall,large_policy,reward_reweight,weight_sweep,pirates_2}"
export NOTES="${NOTES:-Pirates_2 large policy with recall reward p095/n005 and strict any wrong-key penalty.}"
export POLICY_PI_ARCH="${POLICY_PI_ARCH:-2048,1024,512}"
export POLICY_VF_ARCH="${POLICY_VF_ARCH:-2048,1024,512}"
export MIMIC_TASK="${MIMIC_TASK:-Pirates_2}"
export ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-Pirates_2}"
export SEED="${SEED:-42}"

run_ppo_petrunko "large_policy_recall_reward_p095_n005_pirates_2" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.95}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.05}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-any}"
