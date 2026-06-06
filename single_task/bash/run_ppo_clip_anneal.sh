#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export CLIP_RANGE="${CLIP_RANGE:-0.30}"

run_ppo_petrunko "clip_anneal" \
    --clip-range-final "${CLIP_RANGE_FINAL:-0.08}" \
    --target-kl "${TARGET_KL:-0.05}"
