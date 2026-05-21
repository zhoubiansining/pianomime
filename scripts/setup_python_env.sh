#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_PROJECT_DIR/configs/baseline.toml}"
eval "$("${CONFIG_PYTHON:-python3}" "$SCRIPT_PROJECT_DIR/scripts/config_export.py" "$CONFIG_FILE" paths)"
PYTHON="${PYTHON:-python3}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

log "Project: $PROJECT_DIR"
log "Virtualenv: $VENV"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -m pip install --upgrade pip "setuptools<81" wheel

if ! python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
then
  log "Installing CUDA 11.8 PyTorch wheels"
  python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118
fi

if ! python -m pip install -r "$PROJECT_DIR/requirements.txt"; then
  log "Full requirements install failed; retrying without optional audio bindings"
  tmp_req="$(mktemp)"
  grep -Ev '^(PyAudio|pyFluidSynth)(==.*)?$' "$PROJECT_DIR/requirements.txt" > "$tmp_req"
  python -m pip install -r "$tmp_req"
  rm -f "$tmp_req"
fi

python - <<'PY'
import torch
print("python env ready")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
PY
