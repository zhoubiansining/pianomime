#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ppo_petrunko_common.sh"

export POLICY_PI_ARCH="${POLICY_PI_ARCH:-2048,1024,512}"
export POLICY_VF_ARCH="${POLICY_VF_ARCH:-1024,256}"

run_ppo_petrunko "actor_large"
