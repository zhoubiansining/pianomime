# PianoMime Code Audit

Current detailed change list: `docs/CODE_MODIFICATION_SUMMARY.md`.

Current reproduced result list: `docs/BASELINE_RESULTS.md`.

Updated: 2026-05-13

## Overall Code Logic

This repository has two separate baseline paths.

### Single-Song PPO Residual Policy

- Entry point: `single_task/train_ppo.py`.
- Evaluation/replay helper: `single_task/test_trained_actions.py`.
- Environment construction: `single_task/utils.py`.
- Data source:
  - `dataset/notes/{song}.pkl`
  - `dataset/high_level_trajectories/{song}_left_hand_action_list.npy`
  - `dataset/high_level_trajectories/{song}_right_hand_action_list.npy`
  - `dataset/low_level_policies/{song}/actions_{song}.npy` for provided action replay.
- Important interpretation:
  - The prior action is produced from the external fingertip demonstration trajectory through IK/QP.
  - PPO learns a residual correction on top of that prior action.
  - This path does not load high-level or low-level diffusion checkpoints.

### Generalist Diffusion Policy

- High-level entry point: `multi_task/eval_high_level.py`.
- Low-level entry point: `multi_task/eval_low_level.py`.
- Shared environment/observation helpers: `multi_task/utils.py`.
- Data/checkpoints:
  - `dataset_hl.zarr`
  - `dataset_ll.zarr`
  - `checkpoint_ae.ckpt`
  - `checkpoint_high_level.ckpt`
  - `checkpoint_low_level.ckpt`
- Important interpretation:
  - High-level diffusion generates fingertip trajectories from MIDI/goal.
  - Low-level diffusion generates robot actions from observations and high-level trajectories.
  - This path does not use single-song PPO checkpoints.

## Code Issues Found

- Several scripts were CWD-dependent and assumed they were launched from one specific parent directory.
- Shell scripts hard-coded GPU 0 and `MUJOCO_EGL_DEVICE_ID=0`, making shared-server usage brittle.
- Multi-task observation helpers used global `.cuda()` instead of the model's actual device.
- Encoder calls inside observation construction were not wrapped in `torch.no_grad()`, causing unnecessary graph construction during evaluation.
- CLI argument checks happened after heavy imports, so even invalid commands loaded MuJoCo/RoboPianist/diffusion dependencies before failing.
- `multi_task/utils.py` imported RoboPianist/MuJoCo environment dependencies at module import time, even when callers only needed lightweight observation helpers.
- Some legacy code paths still reference old `handtracking/...` paths or unused training scripts; these should be cleaned carefully after baseline reproduction.
- The project still lacks a reliable lockfile or tested one-command environment setup.

## Changes Made On 2026-05-13

All changes below are engineering/runtime hygiene changes. They do not change model architectures, reward functions, checkpoint contents, diffusion scheduler settings, PPO hyperparameters, or environment task definitions.

- `multi_task/utils.py`
  - Added repo-root path helpers.
  - Replaced global `.cuda()` usage in `get_diffusion_obs()` with device selection from the actual encoder/plan-encoder module.
  - Wrapped encoder/plan-encoder forward calls in `torch.no_grad()` during observation construction.
  - Replaced mutable default list arguments with `None`.
  - Replaced broad train/test note loading `try/except` blocks with an explicit helper that checks `dataset/notes` first and falls back to `dataset/notes_test`.
  - Made RoboPianist/MuJoCo environment imports lazy so importing utility functions is lighter.
  - Kept the same numerical observation construction and environment parameters.

- `multi_task/eval_high_level.py`
  - Replaced fragile `sys.path.append("pianomime")` with repo-root insertion based on `__file__`.
  - Moved missing-argument validation before heavy imports.
  - Removed unused imports and an unused `EMAModel` object.

- `multi_task/eval_low_level.py`
  - Replaced fragile `sys.path.append("pianomime")` with repo-root insertion based on `__file__`.
  - Moved missing-argument validation before heavy imports.
  - Removed unused imports and unused notebook video-display helper.

- `single_task/test_trained_actions.py`
  - Replaced fragile `sys.path.append("pianomime")` with repo-root insertion based on `__file__`.
  - Moved missing-argument validation before heavy imports.
  - Removed unused heavy imports.
  - Kept action replay, wrapper order, metrics, and video recording unchanged.

- `single_task/train_ppo.py`
  - Replaced fragile `sys.path.append("pianomime")` with repo-root insertion based on `__file__`.
  - Kept PPO hyperparameters and environment configuration unchanged.
  - Existing earlier change remains: writes `eval_metrics.csv` and `eval_f1_curve.png` during PPO training.

- `single_task/utils.py`
  - Restored missing imports for latent helper functions that referenced `piano_with_shadow_hands`.
  - Added a small note-trajectory loader using repo-root paths.
  - Kept single-song prior/residual environment logic unchanged.

- Shell scripts:
  - `scripts/run_ppo.sh`
  - `single_task/run_ppo.sh`
  - `scripts/eval_low_level.sh`
  - `scripts/test_trained_actions.sh`
  - Added `set -euo pipefail`.
  - Resolve project root from script location.
  - Allow overriding `PYTHON_BIN`, `RUN_DIR`, `ROOT_DIR`, `CUDA_VISIBLE_DEVICES`, and `MUJOCO_EGL_DEVICE_ID`.
  - Preserve the original command-line hyperparameters.

## Validation

- `py_compile` passed for:
  - `multi_task/utils.py`
  - `multi_task/eval_high_level.py`
  - `multi_task/eval_low_level.py`
  - `single_task/utils.py`
  - `single_task/test_trained_actions.py`
  - `single_task/train_ppo.py`
- Shell syntax checks passed for the updated shell scripts.
- Missing-argument checks now return immediately without heavy imports.
- RoboPianist environment reset/step smoke test passed after the edits.
- Changes were synced to the local runtime copy at `/home/gaoj/piano_scratch/pianomime`.

## Do Not Change Casually

- Reward terms and wrapper order in single-song PPO.
- `residual_factor`, `trim_silence`, `n_steps_lookahead`, `frame_stack`, and other baseline hyperparameters.
- Diffusion `pred_horizon`, `action_horizon`, `obs_horizon`, number of diffusion steps, beta schedule, and checkpoint loading.
- Demonstration trajectory source for single-song PPO.
- Train/test song split and note trajectory lookup order.

## Suggested Next Cleanup

- Create a tested `requirements-repro.txt` or lockfile for Python 3.11 + CUDA 11.8.
- Split `multi_task/utils.py` into observation utilities and environment-construction utilities.
- Remove or archive unused experimental training scripts after baseline reproduction is fully documented.
- Push the current local Git repository after the team creates the empty GitHub remote.

## Changes Made On 2026-05-14

These changes are packaging, automation, and interruption-safety changes only.
They do not change baseline rewards, task definitions, checkpoint loading,
diffusion/PPO hyperparameters, or policy architectures.

- Added project documentation:
  - `COURSE_BASELINE.md`
  - `docs/USAGE.md`
  - `docs/EXPERIMENT_AUTOMATION.md`
  - `docs/4090_FEASIBILITY.md`
  - `docs/CONFIGURATION.md`
  - `docs/BASELINE_RESULTS.md`
  - `docs/CODE_MODIFICATION_SUMMARY.md`
- Added automation scripts:
  - `scripts/setup_python_env.sh`
  - `scripts/setup_artifacts.sh`
  - `scripts/sync_to_runtime.sh`
  - `scripts/check_4090_feasibility.sh`
  - `scripts/baseline_scheduler.sh`
  - `scripts/start_tmux_baseline.sh`
- Updated `.gitignore` to keep downloaded checkpoints, datasets, videos,
  generated trajectories, local run directories, and caches out of Git.
- Updated `single_task/train_ppo.py` for safer reruns:
  - Reuses an existing experiment directory.
  - Preserves an existing `eval_metrics.csv` instead of truncating it.
  - Checks `latest_filename` before deleting generated videos.
  - Skips final rollout cleanly if an interrupted run has not produced a best
    checkpoint yet.

## Validation On 2026-05-14

- Shell syntax check passed for all new automation scripts.
- `py_compile` passed for the modified Python entrypoints and utilities.
