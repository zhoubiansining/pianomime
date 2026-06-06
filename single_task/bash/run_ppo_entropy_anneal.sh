#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export ENT_COEF="${ENT_COEF:-0.02}"

run_ppo_petrunko "entropy_anneal" \
    --ent-coef-final "${ENT_COEF_FINAL:-0.001}"
