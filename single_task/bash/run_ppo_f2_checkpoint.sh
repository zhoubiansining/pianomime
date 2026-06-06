#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Petrunko_3-recall}"
export TAGS="${TAGS:-recall,f2_checkpoint}"
export NOTES="${NOTES:-Keep baseline training unchanged but save the best checkpoint by F2 score so recall is weighted more heavily than precision.}"

run_ppo_petrunko "f2_checkpoint" \
    --checkpoint-metric "${CHECKPOINT_METRIC:-fbeta}" \
    --checkpoint-f-beta "${CHECKPOINT_F_BETA:-2.0}"
