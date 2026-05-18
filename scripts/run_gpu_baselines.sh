#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entrypoint kept for older notes. The actual task list,
# paths, GPU polling threshold, and baseline hyperparameters now live in
# configs/baseline.toml and are consumed by baseline_scheduler.sh.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"

exec bash "$PROJECT_ROOT/scripts/baseline_scheduler.sh"
