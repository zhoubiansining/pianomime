#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export VF_COEF="${VF_COEF:-0.75}"

run_ppo_petrunko "value_clip" \
    --clip-range-vf "${CLIP_RANGE_VF:-0.20}"
