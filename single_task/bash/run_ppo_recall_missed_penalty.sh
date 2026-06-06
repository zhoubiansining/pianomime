#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export WANDB_PROJECT="${WANDB_PROJECT:-robopianist-Petrunko_3-recall}"
export TAGS="${TAGS:-recall,missed_note_penalty}"
export NOTES="${NOTES:-Penalize inactive target keys directly to make missed notes more expensive than in the baseline reward.}"

run_ppo_petrunko "recall_missed_penalty" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.7}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.3}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-fraction}" \
    --missed-note-penalty-coef "${MISSED_NOTE_PENALTY_COEF:-0.2}"
