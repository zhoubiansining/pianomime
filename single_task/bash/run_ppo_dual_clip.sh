#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

run_ppo_petrunko "dual_clip" \
    --dual-clip-coef "${DUAL_CLIP_COEF:-3.0}"
