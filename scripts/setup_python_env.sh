#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT="${SHARED_ROOT:-/home/gaoj/share4/_piano}"
PROJECT_DIR="${PROJECT_DIR:-$SHARED_ROOT/pianomime}"
VENV="${VENV:-$SHARED_ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

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
python -m pip install --upgrade pip setuptools wheel

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
