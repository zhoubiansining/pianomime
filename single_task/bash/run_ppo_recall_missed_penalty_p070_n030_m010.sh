#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export TOTAL_ITERS="${TOTAL_ITERS:-2000}"
export TAGS="${TAGS:-recall,missed_note_penalty,tune,precision_recovery}"
export NOTES="${NOTES:-Tune missed-note penalty: positive=0.70, negative=0.30, missed_penalty=0.10.}"

run_ppo_petrunko "recall_missed_penalty_p070_n030_m010" \
    --key-press-positive-weight "${KEY_PRESS_POSITIVE_WEIGHT:-0.70}" \
    --key-press-negative-weight "${KEY_PRESS_NEGATIVE_WEIGHT:-0.30}" \
    --key-press-negative-mode "${KEY_PRESS_NEGATIVE_MODE:-fraction}" \
    --missed-note-penalty-coef "${MISSED_NOTE_PENALTY_COEF:-0.10}"
