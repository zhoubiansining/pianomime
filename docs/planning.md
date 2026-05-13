# PianoMime Baseline Reproduction Plan

Updated: 2026-05-12

## Course Requirement Extract

- Track: Dexterous Piano Track.
- Baseline task:
  - Run PianoMime original repo experiments.
  - Select 3 songs from the training dataset that sound best for the robot to play.
  - Visualize the F1 score training curve.
  - Produce final performance videos.
  - Also train a multi-song policy, or use provided checkpoints, and show performance video on unseen test songs.

## Current Reproduction Strategy

1. Inspect the PDF and repository structure.
2. Verify whether local code can run before changing algorithmic behavior.
3. Install missing runtime dependencies in a project-local environment if possible.
4. Recover or download the missing dataset/checkpoint artifacts required by the original scripts.
5. Run smoke tests for:
   - RoboPianist environment construction.
   - single-song PPO training/evaluation entrypoints.
   - multi-task high-level and low-level checkpoint evaluation entrypoints.
6. Produce baseline outputs under a dedicated results directory:
   - training logs/curves,
   - final F1/precision/recall summaries,
   - videos,
   - exact commands and configs.
7. Refactor only minimal code needed for reliable reproduction; document larger cleanup tasks separately.

## Notes So Far

- PDF has 16 pages; Dexterous Piano Track requirement is on pages 6-8.
- Local repo path: `/home/gaoj/share4/_piano/pianomime`.
- The repo currently is not a Git repository: no `.git` directory was found under `pianomime`.
- Local Python is `/home/gaoj/anaconda3/bin/python`, version 3.11.5.
- Existing Python environment has `torch==2.1.0` and `gymnasium==0.29.1`, but most PianoMime dependencies are missing.
- Local repo includes `dataset_ll.zarr` and `dataset_hl.zarr`, but not the full `dataset/` folder expected by the single-task and multi-task evaluation code.

## Completed So Far

- [x] Created a project-local environment at `/home/gaoj/share4/_piano/.venv`.
- [x] Installed the runtime dependencies needed to import and run the main RoboPianist/PianoMime code paths.
- [x] Downloaded and extracted the official PianoMime dataset and checkpoints.
  - Dataset folders restored under `/home/gaoj/share4/_piano/pianomime/dataset`.
  - Checkpoints restored under `/home/gaoj/share4/_piano/pianomime`.
- [x] Patched fragile imports, output-directory creation, CPU/GPU fallback, and hard-coded path assumptions.
- [x] Confirmed RoboPianist can construct, reset, and step a task under MuJoCo EGL.
- [x] Replayed 3 training-song single-task baselines and produced metrics/videos.
- [x] Ran multi-song high-level and low-level checkpoint evaluation for unseen test song `Alone_1`.
- [x] Patched `single_task/train_ppo.py` to save `eval_metrics.csv` and `eval_f1_curve.png` during PPO training.

## Baseline Outputs

Results are being collected under `/home/gaoj/share4/_piano/baseline_results`.

Important interpretation note:

- Single-song policy and generalist policy are separate baselines in this repo.
- Single-song policy path: `single_task/train_ppo.py`.
  - It uses one song's existing demonstration trajectory (`left_hand_action_list`, `right_hand_action_list`) plus IK/QP to build `prior_action`.
  - PPO learns a residual correction on top of that prior.
  - It does not load or depend on the high-level/low-level diffusion checkpoints.
- Generalist policy path: `multi_task/eval_high_level.py` plus `multi_task/eval_low_level.py`.
  - High-level diffusion generates fingertip trajectories from MIDI/goal.
  - Low-level diffusion generates actions from those trajectories/observations.
  - It does not use single-song PPO checkpoints.
- Therefore, single-song results should be reported as demonstration-conditioned PPO residual results, while multi-song results should be reported as generalist diffusion checkpoint results.

Single-song replay results:

| Song | Split | Precision | Recall | F1 | Video |
| --- | --- | ---: | ---: | ---: | --- |
| `Stan_1` | train | 0.9991 | 0.9719 | 0.9795 | `single_song/videos/Stan_1_single_song_baseline.mp4` |
| `Petrunko_3` | train | 0.9869 | 0.8460 | 0.8900 | `single_song/videos/Petrunko_3_single_song_baseline.mp4` |
| `NeverGonnaGiveYouUp_1` | train | 0.9960 | 0.9260 | 0.9514 | `single_song/videos/NeverGonnaGiveYouUp_1_single_song_baseline.mp4` |

Multi-song checkpoint evaluation:

- `Alone_1` high-level stage completed and generated:
  - `multi_task/trajectories/Alone_1_trajectory.npy`
  - `multi_task/trajectories/Alone_1_left_hand_action_list.npy`
  - `multi_task/trajectories/Alone_1_right_hand_action_list.npy`
- `Alone_1` low-level stage completed:
  - Precision: 0.8283
  - Recall: 0.6443
  - F1: 0.7902
  - Video: `multisong/videos/Alone_1_multisong_baseline.mp4`

## Remaining TODO

- [x] Finish `Alone_1` multi-song low-level evaluation and copy the final video/metrics into `baseline_results/multisong`.
- [ ] GPU watcher is running and will start the next experiments automatically.
  - Scheduler PID: `3280417`.
  - Script: `/home/gaoj/share4/_piano/run_gpu_baselines.sh`.
  - Log: `/home/gaoj/piano_scratch/baseline_results/gpu_scheduler_20260513_135059.log`.
  - Current state: GPUs 5/6/7 were no longer idle at launch time; they are occupied by user `wuwl`'s `dinov3` jobs, so the watcher is waiting.
- [ ] Re-run the additional unseen test song `Numb_1`.
  - Assigned to GPU 5 once free.
  - First attempt reached shared-filesystem wait (`rpc_wait_bit_killable`) during high-level startup and was stopped before producing a trajectory.
- [ ] Run another unseen test song `NoTimeToDie_1`.
  - Assigned to GPU 6 once free.
- [ ] Produce the requested PPO F1 training curve.
  - `Petrunko_3` PPO training is assigned to GPU 7 once free.
  - The provided checkpoints do not include training logs.
  - Re-running full PPO training is possible but expensive and should ideally use a CUDA-enabled PyTorch environment.
  - `train_ppo.py` now persists curve data for future full runs.
- [ ] Clean up dependency documentation once the final runnable environment is stable.
- [ ] Turn the local code edits into a proper Git commit once the repo is re-initialized or restored as a Git checkout.

## Maintenance Notes

- Code audit and change log: `/home/gaoj/share4/_piano/code_audit.md`.
- Replace remaining hard-coded relative paths with `PROJECT_ROOT`-relative paths.
- Replace ad-hoc `sys.path.append("pianomime")` imports with package-relative imports or a proper editable install.
- Move shell-script CUDA/GPU assumptions into documented environment variables.
- Update dependency pins and create a reproducible environment file.
- Reduce noisy debug `print()` calls in library/dataset code; keep progress reporting in CLI entrypoints.

## 2026-05-14 Update

- [x] Moved the deliverable documentation into the project itself:
  - `/home/gaoj/share4/_piano/pianomime/COURSE_BASELINE.md`
  - `/home/gaoj/share4/_piano/pianomime/docs/USAGE.md`
  - `/home/gaoj/share4/_piano/pianomime/docs/EXPERIMENT_AUTOMATION.md`
  - `/home/gaoj/share4/_piano/pianomime/docs/4090_FEASIBILITY.md`
  - `/home/gaoj/share4/_piano/pianomime/docs/GITHUB_RELEASE.md`
  - `/home/gaoj/share4/_piano/pianomime/docs/CODEX_HANDOFF_PROMPT.md`
- [x] Added tmux-based automation that keeps running after VSCode/Codex disconnects:
  - `scripts/start_tmux_baseline.sh`
  - `scripts/baseline_scheduler.sh`
- [x] Added shared-directory-aware setup scripts:
  - `scripts/setup_python_env.sh`
  - `scripts/setup_artifacts.sh`
  - `scripts/sync_to_runtime.sh`
  - `scripts/check_4090_feasibility.sh`
- [x] Added a proper `.gitignore` so large downloaded/generated artifacts do not get committed.
- [x] Made PPO training safer to resume/re-run:
  - Experiment directory creation now uses `exist_ok=True`.
  - `eval_metrics.csv` is not overwritten if it already exists.
  - Video cleanup checks that the file exists.
  - Final rollout is skipped gracefully if an interrupted run produced no best checkpoint.
- [ ] Start the new tmux scheduler and retire the old non-tmux watcher to avoid duplicate launches.
- [ ] Initialize a local Git repo and commit the cleaned project.
- [ ] Push to GitHub after the user creates/provides the empty remote repository URL.

## 2026-05-14 Runtime Update

- [x] Existing non-tmux watcher successfully started the queued jobs when GPUs 5/6/7 became free:
  - `Numb_1` multi-song eval
  - `NoTimeToDie_1` multi-song eval
  - `Petrunko_3` PPO training curve
- [x] Created tmux session `pianomime_baseline` so progress can be monitored after disconnect:
  - Attach with `tmux attach -t pianomime_baseline`.
- [x] Fixed a lazy-import regression in `multi_task/utils.py`:
  - `DeepMimicWrapper`, `FingeringEmbWrapper`, and `ResidualWrapper` are now imported through `_env_deps()`.
  - The fix was synced to `/home/gaoj/piano_scratch/pianomime`.
- [x] Restarted failed low-level evals for `Numb_1` and `NoTimeToDie_1` in tmux without touching the ongoing PPO run.
- [ ] Let the original `Petrunko_3` PPO run continue; current progress is recorded in `eval_metrics.csv`.
- [ ] Confirm final `Numb_1` and `NoTimeToDie_1` F1/video outputs after the low-level reruns complete.

## 2026-05-14 A800 Batch

- [x] Confirmed GPUs 4/5/6/7 became idle.
- [x] Confirmed original queued baseline finished:
  - `Numb_1` generalist: F1 0.7504246031746031.
  - `NoTimeToDie_1` generalist: F1 0.8552746790246791.
  - `Petrunko_3` PPO: 2000 iterations completed; final reported F1 0.7956861471861472.
- [x] Started four additional unseen-song generalist evals in tmux:
  - `Forester_1` on GPU 4.
  - `SomewhereOnlyWeKnow_1` on GPU 5.
  - `EyesClosed_1` on GPU 6.
  - `Paradise_1` on GPU 7.
- [x] Added direct per-song runner `scripts/run_multisong_task.sh` to avoid scheduler startup rsync.
- [x] Compiled and launched CUDA memory guards because Python startup was temporarily blocked on shared/NFS reads before CUDA allocation:
  - Binary: `/home/gaoj/piano_scratch/gpu_guard`.
  - Source recorded in `scripts/gpu_guard.cu`.
  - Guards reserve about 32 GiB on each of GPUs 4/5/6/7, leaving about 48 GiB for the actual evals.
  - Release command: `touch /home/gaoj/piano_scratch/stop_gpu_guard_a800_batch_20260514`.
- [x] Confirmed Python evals left `wait_on_page_bit_common` / `rpc_wait_bit_killable`, completed, and wrote metrics/videos.
- [x] Released GPU guards after completion; GPUs 4/5/6/7 returned to about 3 MiB used.
- [ ] If future jobs again spend minutes in NFS waits, build or copy a fully local Python runtime to reduce dependence on `/home/gaoj/anaconda3` and shared `.venv`.

Additional A800 batch results:

| Song | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `Forester_1` | 0.8115715162676489 | 0.7300491098833639 | 0.7944005963342979 |
| `SomewhereOnlyWeKnow_1` | 0.6516388888888889 | 0.5788888888888889 | 0.7920031265031265 |
| `EyesClosed_1` | 0.6126666666666666 | 0.5150555555555555 | 0.8569007936507936 |
| `Paradise_1` | 0.8392460317460317 | 0.7535 | 0.8104489214489214 |
