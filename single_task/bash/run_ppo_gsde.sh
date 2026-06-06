#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export SDE_SAMPLE_FREQ="${SDE_SAMPLE_FREQ:-16}"
export LOG_STD_INIT="${LOG_STD_INIT:--0.5}"

run_ppo_petrunko "gsde" \
    --use-sde
