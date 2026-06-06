#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export ACTIVATION_FN="${ACTIVATION_FN:-silu}"
export LOG_STD_INIT="${LOG_STD_INIT:--0.5}"

run_ppo_petrunko "silu_no_ortho" \
    --no-ortho-init
