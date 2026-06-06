#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Petrunko_3-recall}"
export TAGS="${TAGS:-recall,reward_reweight}"
export NOTES="${NOTES:-Increase target-key reward weight and reduce wrong-key avoidance weight to trade some precision margin for higher recall.}"

run_ppo_petrunko "recall_reward" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.8}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.2}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-any}"
