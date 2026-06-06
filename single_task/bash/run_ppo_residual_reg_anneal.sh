#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

run_ppo_petrunko "residual_reg_anneal" \
    --residual-action-regularization \
    --residual-l2-coef "${RESIDUAL_L2_COEF:-5e-4}" \
    --residual-l2-coef-final "${RESIDUAL_L2_COEF_FINAL:-0.0}" \
    --residual-smooth-coef "${RESIDUAL_SMOOTH_COEF:-5e-3}" \
    --residual-smooth-coef-final "${RESIDUAL_SMOOTH_COEF_FINAL:-0.0}"
