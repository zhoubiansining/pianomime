#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

run_ppo_petrunko "residual_reg" \
    --residual-action-regularization \
    --residual-l2-coef "${RESIDUAL_L2_COEF:-1e-3}" \
    --residual-smooth-coef "${RESIDUAL_SMOOTH_COEF:-1e-2}"
