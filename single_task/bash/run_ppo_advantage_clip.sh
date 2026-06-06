#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

run_ppo_petrunko "advantage_clip" \
    --advantage-clip "${ADVANTAGE_CLIP:-5.0}"
