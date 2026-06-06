#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export N_STEPS="${N_STEPS:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export GAE_LAMBDA="${GAE_LAMBDA:-0.98}"

run_ppo_petrunko "long_horizon"
