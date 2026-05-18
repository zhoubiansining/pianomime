# Usage

This document describes the course-maintained baseline workflow. Paths assume
the shared server directory is mounted at `/home/gaoj/share4`.

## Directory Layout

```text
/home/gaoj/share4/_piano/
  pianomime/                 # shared source tree
  .venv/                     # shared Python environment, if available
  artifacts/                 # cached Google Drive zips
  baseline_results/          # synced logs, metrics, videos, training outputs

/home/gaoj/piano_scratch/
  pianomime/                 # local runtime copy, used to avoid slow shared I/O
  baseline_results/          # local result staging area
  runs/                      # local run directories
```

The shared source tree is the source of truth. Long-running experiments should
run from the local runtime copy created by `scripts/sync_to_runtime.sh`.

For the already reproduced baseline metrics, videos, and training curve, read
`docs/BASELINE_RESULTS.md` first. Do not rerun completed baseline jobs unless
you are validating a new environment.

## Environment

Use the existing environment if it is present:

```bash
source /home/gaoj/share4/_piano/.venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

If the environment is missing, rebuild it:

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_python_env.sh
```

If network access fails, export one of the provided proxies first:

```bash
export http_proxy=http://10.0.0.204:1080 https_proxy=http://10.0.0.204:1080
# or
export http_proxy=http://10.0.0.204:1090 https_proxy=http://10.0.0.204:1090
```

## Artifacts

Prepare dataset and checkpoints:

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_artifacts.sh
```

The script reuses cached zips from `/home/gaoj/share4/_piano/artifacts` when
available. It downloads from the original PianoMime Google Drive links only if
the cache is missing.

## Smoke Checks

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/check_4090_feasibility.sh
```

Despite the name, this is a general CUDA/RoboPianist smoke test. On a 4090 it
checks that CUDA allocation works and that the project imports in headless mode.

## Manual Baseline Commands

Single-song action replay:

```bash
cd /home/gaoj/piano_scratch/runs/single_Stan_1
export PYTHONPATH=/home/gaoj/piano_scratch/pianomime:/home/gaoj/piano_scratch/pianomime/single_task
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl
/home/gaoj/share4/_piano/.venv/bin/python \
  /home/gaoj/piano_scratch/pianomime/single_task/test_trained_actions.py Stan_1
```

Generalist diffusion baseline:

```bash
cd /home/gaoj/piano_scratch/runs/multisong_Alone_1
export PYTHONPATH=/home/gaoj/piano_scratch/pianomime
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl
/home/gaoj/share4/_piano/.venv/bin/python \
  /home/gaoj/piano_scratch/pianomime/multi_task/eval_high_level.py Alone_1
/home/gaoj/share4/_piano/.venv/bin/python \
  /home/gaoj/piano_scratch/pianomime/multi_task/eval_low_level.py Alone_1
```

PPO residual baseline uses the original task hyperparameters through:

```bash
bash scripts/run_ppo.sh Petrunko_3
```

For unattended runs, prefer the tmux scheduler described in
`docs/EXPERIMENT_AUTOMATION.md`.

Current reproduced result files are under:

```bash
ls /home/gaoj/share4/_piano/baseline_results/single_song/videos
ls /home/gaoj/share4/_piano/baseline_results/multisong/videos
```
