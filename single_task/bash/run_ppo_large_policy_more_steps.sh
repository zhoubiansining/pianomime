#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export TOTAL_ITERS="${TOTAL_ITERS:-3000}"
export MAX_STEPS="${MAX_STEPS:-50000000}"
export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Petrunko_3-steps-compare}"
export TAGS="${TAGS:-steps_compare,large_policy,more_steps}"
export NOTES="${NOTES:-Petrunko_3 large_policy extended from 2000 to 3000 PPO iterations for late-training comparison.}"
export POLICY_PI_ARCH="${POLICY_PI_ARCH:-2048,1024,512}"
export POLICY_VF_ARCH="${POLICY_VF_ARCH:-2048,1024,512}"

run_ppo_petrunko "large_policy_more_steps"
