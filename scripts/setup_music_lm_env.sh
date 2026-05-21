#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$PROJECT_ROOT/scripts/config_export.py" "$CONFIG_FILE" music_lm)"

PYTHON="${PYTHON:-python3}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

log() {
  printf '[music-lm-env] %s\n' "$*"
}

if [[ ! -x "$MUSIC_LM_VENV/bin/python" ]]; then
  log "Creating virtualenv at $MUSIC_LM_VENV"
  "$PYTHON" -m venv "$MUSIC_LM_VENV"
fi

# shellcheck source=/dev/null
source "$MUSIC_LM_VENV/bin/activate"
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -r "$PROJECT_ROOT/requirements_music_lm.txt"

python - <<'PY'
import torch
import note_seq
import pretty_midi
print("music_lm env ready")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("note_seq", getattr(note_seq, "__version__", "ok"))
print("pretty_midi", pretty_midi.__version__)
PY
