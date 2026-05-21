# Code Modification Summary

Last updated: 2026-05-19

This document records the engineering changes made to make the PianoMime
baseline reproducible on the course server. The intent was to preserve the
original baseline algorithms while making the repository runnable, resumable,
and easier for teammates to maintain.

## Safety Boundary

The following baseline semantics were intentionally kept unchanged:

- No reward terms were redesigned.
- No model architecture was changed.
- No diffusion checkpoint or task definition was modified.
- The default PPO policy architecture and core baseline hyperparameters remain
  unchanged; the defaults are now centralized in `configs/baseline.toml` for
  future copied experiment configs.

Most edits are path handling, dependency tolerance, logging, automation, and
runtime stability improvements.

## Modified Areas

| Area | Files | What changed |
| --- | --- | --- |
| Multi-task diffusion eval | `multi_task/eval_high_level.py`, `multi_task/eval_low_level.py` | Repo-root path resolution, early CLI validation, device-aware checkpoint loading, output directory creation. |
| Multi-task utilities | `multi_task/utils.py` | Lazy environment imports, repo-relative dataset lookup, CPU/GPU-safe encoder device use, `torch.no_grad()` in eval observation encoding. |
| Single-song PPO training | `single_task/train_ppo.py` | Persistent evaluation CSV, F1 curve image, safer reruns, optional pretrained resume, final-rollout guard. |
| Single-song replay | `single_task/test_trained_actions.py`, `single_task/utils.py` | Repo-relative dataset/action loading and lighter startup validation. |
| Central config | `configs/baseline.toml`, `pianomime_config.py`, `scripts/config_export.py`, `scripts/run_ppo_from_config.py` | Centralized paths, task lists, scheduler defaults, and single-song/generalist baseline hyperparameters in TOML. |
| Audio/video robustness | `robopianist/wrappers/sound.py` | Audio dependencies are optional at runtime; missing FluidSynth/PortAudio no longer prevents silent video generation. |
| Automation scripts | `scripts/*.sh`, especially `baseline_scheduler.sh`, `start_tmux_baseline.sh`, `sync_to_runtime.sh`, `run_multisong_task.sh` | tmux execution, GPU waiting/locking, scratch-directory execution, result syncing, rerunnable tasks. |
| Setup/release docs | `README.md`, `COURSE_BASELINE.md`, `docs/*.md`, `.gitignore`, `requirements.txt` | Added course documentation, artifact setup, result index, code audit, and ignored large generated/downloaded artifacts. |

## Representative Changes

### 1. Central config

Files:

```text
configs/baseline.toml
pianomime_config.py
scripts/config_export.py
scripts/run_ppo_from_config.py
scripts/baseline_scheduler.sh
scripts/run_ppo.sh
single_task/train_ppo.py
single_task/test_trained_actions.py
multi_task/eval_high_level.py
multi_task/eval_low_level.py
```

Baseline paths, task lists, and core hyperparameters now come from
`configs/baseline.toml`:

```toml
[single_song.ppo]
total_iters = 2000
residual_factor = 0.03
ppo_batch_size = 1024
policy_activation = "gelu"
policy_pi_arch = [1024, 256]
policy_vf_arch = [1024, 256]
```

Entry scripts use the same config loader:

```python
cfg = load_config(args.config)
ppo = dict(section(cfg, "single_song", "ppo"))
command.extend(cli_args_from_mapping(ppo))
```

Future methods can copy a config rather than chasing defaults across shell and
Python files.

### 2. Repo-root path handling

Files:

```text
multi_task/eval_high_level.py
multi_task/eval_low_level.py
single_task/train_ppo.py
single_task/test_trained_actions.py
single_task/utils.py
```

The original scripts depended on being launched from a particular parent
directory. They now resolve the repository root from `__file__`.

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Representative artifact loading now uses `PROJECT_ROOT`:

```python
dataset_path = str(PROJECT_ROOT / "dataset_ll.zarr")
ckpt_path = PROJECT_ROOT / "checkpoint_low_level.ckpt"
left_hand_action_list = np.load(
    PROJECT_ROOT / "multi_task" / "trajectories" / f"{task_name}_left_hand_action_list.npy"
)
```

This makes the scripts usable from tmux run directories under local scratch,
instead of requiring the user to manually `cd` into one specific folder.

### 3. CPU/GPU-safe model loading

Files:

```text
multi_task/eval_high_level.py
multi_task/eval_low_level.py
multi_task/utils.py
```

Checkpoint loading now follows the actual available device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
```

Observation encoding no longer assumes `.cuda()`:

```python
def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

device = _module_device(encoder)
goal = torch.from_numpy(goal).to(device)
with torch.no_grad():
    goal = encoder.forward_without_sampling(goal)
```

This prevents CPU-only fallback from crashing and avoids building unnecessary
autograd graphs during evaluation.

### 4. Lazy environment imports

File:

```text
multi_task/utils.py
```

Heavy MuJoCo/RoboPianist imports are delayed until an environment is actually
constructed:

```python
def _env_deps():
    from robopianist.suite.tasks import piano_with_shadow_hands_res
    from dm_env_wrappers import CanonicalSpecWrapper, SinglePrecisionWrapper, DmControlWrapper
    from robopianist.wrappers import PianoSoundVideoWrapper
    from robopianist.wrappers.deep_mimic import DeepMimicWrapper
    from robopianist.wrappers.fingering_emb import FingeringEmbWrapper
    from robopianist.wrappers.residual import ResidualWrapper
    ...
```

This makes lightweight imports and invalid-argument checks faster and less
fragile on headless servers.

### 5. Train/test note trajectory lookup

File:

```text
multi_task/utils.py
```

The generalist evaluation can load either training or test note trajectories
without broad exception swallowing:

```python
def _load_note_trajectory(task_name):
    train_path = PROJECT_ROOT / "dataset" / "notes" / f"{task_name}.pkl"
    test_path = PROJECT_ROOT / "dataset" / "notes_test" / f"{task_name}.pkl"
    path = train_path if train_path.exists() else test_path
    with path.open("rb") as f:
        return pickle.load(f)
```

This is important for unseen-song evaluation, where test songs live in
`dataset/notes_test`.

### 6. PPO training metrics and F1 curve

File:

```text
single_task/train_ppo.py
```

The baseline now leaves a machine-readable F1 trace:

```python
def append_eval_metrics(path: Path, iteration: int, env_steps: int, metrics: dict) -> None:
    row = {
        "iteration": iteration,
        "env_steps": env_steps,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "sustain_precision": metrics.get("sustain_precision"),
        "sustain_recall": metrics.get("sustain_recall"),
        "sustain_f1": metrics.get("sustain_f1"),
    }
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row)).writerow(row)
```

At the end of training or interruption, the curve is plotted:

```python
finally:
    maybe_plot_f1_curve(metrics_path, f1_curve_path)
```

The produced files are:

```text
eval_metrics.csv
eval_f1_curve.png
```

### 7. Safer PPO reruns and interruption behavior

Files:

```text
single_task/train_ppo.py
scripts/baseline_scheduler.sh
```

Experiment directories are reused instead of failing if they already exist:

```python
experiment_dir.mkdir(parents=True, exist_ok=True)
if not metrics_path.exists():
    write_eval_metrics_header(metrics_path)
```

The scheduler can resume from an existing best checkpoint:

```bash
checkpoint_zip="$run_dir/robopianist_rl/ckpts/${run_name}_best.zip"
maybe_pretrained=()
if [[ -f "$checkpoint_zip" ]]; then
  maybe_pretrained=(--pretrained "$checkpoint_zip")
fi
```

If no checkpoint was produced before interruption, final rollout is skipped
cleanly:

```python
if not checkpoint_path.with_suffix(".zip").exists():
    print(f"No checkpoint produced at {checkpoint_path.with_suffix('.zip')}; skip final rollout.")
    return
```

### 8. Silent-video fallback

File:

```text
robopianist/wrappers/sound.py
```

The server lacks the system FluidSynth/PortAudio libraries. The video wrapper
now degrades to silent video instead of failing the whole run:

```python
try:
    from robopianist.music import synthesizer
    self._synth = synthesizer.Synthesizer(sf2_path, sample_rate)
except ImportError as exc:
    self._synth = None
    self._synth_error = exc

def _write_frames(self) -> None:
    super()._write_frames()
    if self._synth is None:
        return
```

This keeps visual baseline videos available even without audio support.

### 9. tmux scheduler and GPU waiting

Files:

```text
scripts/start_tmux_baseline.sh
scripts/baseline_scheduler.sh
scripts/sync_to_runtime.sh
scripts/run_multisong_task.sh
```

The scheduler waits for free GPUs, claims one with a lock directory, and writes
all logs/results into stable shared locations:

```bash
claim_gpu() {
  local label="$1"
  while true; do
    for gpu in $GPU_IDS; do
      used="$(gpu_mem_used "$gpu" || echo 999999)"
      lock="$LOCK_DIR/gpu_${gpu}.lock"
      if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_FREE_MEM_MB )); then
        if mkdir "$lock" 2>/dev/null; then
          printf '%s\n' "$BASHPID" > "$lock/pid"
          printf '%s\n' "$gpu"
          return 0
        fi
      fi
    done
    sleep "$POLL_SECONDS"
  done
}
```

The scheduler runs experiments from local scratch:

```bash
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/gaoj/piano_scratch}"
RUNTIME_DIR="${RUNTIME_DIR:-$RUNTIME_ROOT/pianomime}"
```

This avoids excessive simulation/video I/O on the shared filesystem.

### 10. Artifact setup and Git hygiene

Files:

```text
scripts/setup_artifacts.sh
scripts/setup_python_env.sh
.gitignore
requirements.txt
docs/CONFIGURATION.md
docs/USAGE.md
```

Large generated or downloaded files are intentionally ignored:

```gitignore
dataset/
dataset_hl.zarr/
dataset_ll.zarr/
checkpoint_*.ckpt
baseline_results/
multi_task/trajectories/
*.mp4
```

Artifacts can be restored with:

```bash
bash scripts/setup_artifacts.sh
```

## Validation Performed

- `py_compile` passed for the modified Python entrypoints and utilities.
- Shell syntax checks passed for the automation scripts.
- RoboPianist environment reset/step smoke test passed.
- Three single-song replay videos and metrics were generated.
- `Petrunko_3` PPO training produced `eval_metrics.csv`, `eval_f1_curve.png`,
  and a final rollout video.
- Seven unseen-song generalist evaluations produced metrics and videos.

## Remaining Non-Algorithmic Caveats

- Videos are silent unless system FluidSynth/PortAudio libraries are installed.
- The setup has been verified on this A800 server; the 4090 path has a smoke
  script but has not been physically run on a visible 4090 host here.
- A full GitHub push still needs the empty remote repository URL.
