#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Petrunko_3-recall}"
export TAGS="${TAGS:-recall,activation_bonus}"
export NOTES="${NOTES:-Add direct bonus for activating target keys on top of recall-biased key press reward.}"

run_ppo_petrunko "recall_activation_bonus" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.7}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.3}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-fraction}" \
    --recall-activation-bonus-coef "${RECALL_ACTIVATION_BONUS_COEF:-0.2}"
