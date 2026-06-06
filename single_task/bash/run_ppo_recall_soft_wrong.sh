#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Petrunko_3-recall}"
export TAGS="${TAGS:-recall,soft_wrong_penalty}"
export NOTES="${NOTES:-Use a fraction-based wrong-key penalty so one false positive does not remove the whole negative-side reward.}"

run_ppo_petrunko "recall_soft_wrong" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.7}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.3}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-fraction}"
