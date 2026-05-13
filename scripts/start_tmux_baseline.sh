#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT="${SHARED_ROOT:-/home/gaoj/share4/_piano}"
PROJECT_DIR="${PROJECT_DIR:-$SHARED_ROOT/pianomime}"
SESSION="${SESSION:-pianomime_baseline}"
RUN_ID="${RUN_ID:-baseline_$(date +%Y%m%d)}"
RESULTS_DIR="${RESULTS_DIR:-$SHARED_ROOT/baseline_results}"
LOG_DIR="$RESULTS_DIR/logs"
LOG_FILE="$LOG_DIR/tmux_${RUN_ID}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

cmd=$(
  printf 'cd %q && SHARED_ROOT=%q PROJECT_DIR=%q RUN_ID=%q RESULTS_DIR=%q GPU_IDS=%q GPU_FREE_MEM_MB=%q PPO_TOTAL_ITERS=%q bash scripts/baseline_scheduler.sh 2>&1 | tee -a %q' \
    "$PROJECT_DIR" \
    "$SHARED_ROOT" \
    "$PROJECT_DIR" \
    "$RUN_ID" \
    "$RESULTS_DIR" \
    "${GPU_IDS:-}" \
    "${GPU_FREE_MEM_MB:-5000}" \
    "${PPO_TOTAL_ITERS:-2000}" \
    "$LOG_FILE"
)

tmux new-session -d -s "$SESSION" "$cmd"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Detach with: Ctrl-b then d"
echo "Log: $LOG_FILE"
