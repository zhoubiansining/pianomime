#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export TOTAL_ITERS="${TOTAL_ITERS:-3000}"
export MAX_STEPS="${MAX_STEPS:-50000000}"
export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Petrunko_3-recall}"
export TAGS="${TAGS:-recall,large_policy,more_steps,soft_wrong_penalty,activation_bonus,f2_checkpoint}"
export NOTES="${NOTES:-Combine the late-improving large policy with a recall-biased reward and F2 checkpoint selection.}"
export POLICY_PI_ARCH="${POLICY_PI_ARCH:-2048,1024,512}"
export POLICY_VF_ARCH="${POLICY_VF_ARCH:-2048,1024,512}"

run_ppo_petrunko "large_policy_recall" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.75}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.25}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-fraction}" \
    --recall-activation-bonus-coef "${RECALL_ACTIVATION_BONUS_COEF:-0.15}" \
    --checkpoint-metric "${CHECKPOINT_METRIC:-fbeta}" \
    --checkpoint-f-beta "${CHECKPOINT_F_BETA:-2.0}"
