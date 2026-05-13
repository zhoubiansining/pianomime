#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT="${SHARED_ROOT:-/home/gaoj/share4/_piano}"
PROJECT_DIR="${PROJECT_DIR:-$SHARED_ROOT/pianomime}"
VENV="${VENV:-$SHARED_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv

"$PYTHON_BIN" - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(1)

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total_gb = props.total_memory / 1024**3
    print(f"gpu {i}: {props.name}, capability={props.major}.{props.minor}, total={total_gb:.1f} GiB")
    x = torch.empty((1024, 1024, 64), dtype=torch.float16, device=i)
    del x
torch.cuda.empty_cache()
print("CUDA allocation smoke test passed")
PY

"$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

project = Path(os.environ["PYTHONPATH"].split(":")[0])
sys.path.insert(0, str(project))
import robopianist
print("robopianist import passed:", robopianist.__file__)
PY
