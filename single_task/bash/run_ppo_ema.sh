#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

run_ppo_petrunko "ema" \
    --policy-ema-decay "${POLICY_EMA_DECAY:-0.995}" \
    --eval-with-ema
