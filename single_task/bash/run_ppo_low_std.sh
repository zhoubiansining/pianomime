#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export LOG_STD_INIT="${LOG_STD_INIT:--1.0}"

run_ppo_petrunko "low_std" \
    --log-std-min "${LOG_STD_MIN:--2.5}" \
    --log-std-max "${LOG_STD_MAX:--0.3}"
