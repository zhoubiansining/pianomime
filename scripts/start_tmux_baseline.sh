#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_PROJECT_DIR/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$SCRIPT_PROJECT_DIR/scripts/config_export.py" "$CONFIG_FILE" paths environment scheduler)"

RUN_ID="${RUN_ID:-baseline_$(date +%Y%m%d)}"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_PROJECT_DIR}"
LOG_DIR="$RESULTS_DIR/logs"
LOG_FILE="$LOG_DIR/tmux_${RUN_ID}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

cmd=$(
  printf 'cd %q && CONFIG_FILE=%q RUN_ID=%q GPU_IDS=%q GPU_FREE_MEM_MB=%q bash scripts/baseline_scheduler.sh 2>&1 | tee -a %q' \
    "$PROJECT_DIR" \
    "$CONFIG_FILE" \
    "$RUN_ID" \
    "${GPU_IDS:-}" \
    "${GPU_FREE_MEM_MB:-5000}" \
    "$LOG_FILE"
)

tmux new-session -d -s "$SESSION" "$cmd"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Detach with: Ctrl-b then d"
echo "Log: $LOG_FILE"
