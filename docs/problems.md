# PianoMime Problems Log

Updated: 2026-05-12

## Confirmed Problems

1. `pianomime` is not currently a Git repository.
   - Symptom: `git status` from `/home/gaoj/share4/_piano/pianomime` fails with `fatal: not a git repository`.
   - Impact: hard to audit changes, compare with upstream, or coordinate later team maintenance.

2. Required Python dependencies are missing from the current environment.
   - Missing imports observed: `mujoco`, `dm_control`, `stable_baselines3`, `zarr`, `note_seq`, `tyro`, `wandb`, `diffusers`, `mujoco_utils`, `dm_env_wrappers`, `qpsolvers`, `quadprog`.
   - Current workaround: a project-local environment has been created at `/home/gaoj/share4/_piano/.venv`.
   - Remaining issue: `requirements.txt` is still not an accurate one-command environment spec.
   - Impact: baseline scripts cannot start from a fresh environment without the extra pinned packages/workarounds.

3. The local artifact set is incomplete for the scripts as written.
   - Present: `dataset_ll.zarr`, `dataset_hl.zarr`, tutorial sample files, RoboPianist assets.
   - Missing so far: `dataset/notes`, `dataset/notes_test`, `dataset/high_level_trajectories`, `dataset/low_level_policies`, `checkpoint_ae.ckpt`, `checkpoint_high_level.ckpt`, `checkpoint_low_level.ckpt`.
   - Current workaround: official dataset/checkpoint artifacts were downloaded and extracted.
   - Impact: fixed for the current workspace, but future clones need documented artifact setup.

4. Several scripts assume being launched from `/home/gaoj/share4/_piano`, not from the repo root.
   - Example: `scripts/run_ppo.sh` calls `python pianomime/single_task/train_ppo.py`.
   - Current workaround: key multi-task eval scripts now resolve paths relative to the project root.
   - Impact: some remaining scripts are still fragile if launched from an unexpected working directory.

5. The available PyTorch installation is CPU-only even though the server has A800 GPUs.
   - Symptom: `torch.cuda.is_available()` is false in the current environment.
   - Current status: fixed in `/home/gaoj/share4/_piano/.venv` by installing `torch==2.1.0+cu118`.
   - Remaining caveat: GPU availability is still a cluster scheduling/resource issue.
   - Impact: code can now use CUDA, but jobs must wait for free GPUs.

6. Audio rendering dependencies are missing at the system level.
   - `pyaudio` cannot be installed without PortAudio headers.
   - `fluidsynth` Python package imports, but the FluidSynth C library is not available.
   - Current workaround: video rendering now degrades to silent video instead of failing.
   - Impact: generated videos are suitable for visual evaluation, but not audio playback.

7. Version conflicts in the Python stack are non-trivial.
   - `zarr==2.16.1` needs an older `numcodecs`; current workaround pins `numcodecs==0.11.0`.
   - `diffusers==0.11.1` needs older compatible `accelerate`, `transformers`, and `huggingface-hub`.
   - `quadprog==0.1.11` failed on Python 3.11; current workaround uses `quadprog==0.1.13`.
   - Impact: the repo should get a lockfile or tested setup script before teammates build on it.

8. Some source imports were broken by unused legacy dependencies.
   - `single_task/utils.py` imported unused `orbax`/Flax checkpoint helpers.
   - `single_task/eval_ppo.py` imported unused `sac`, `specs`, `replay`, `orbax`, and `flax` modules.
   - Current workaround: removed the unused imports from those files.

9. Training-curve requirement is not yet satisfied.
   - The provided checkpoints do not include F1 training logs.
   - The current collected results are checkpoint/action replay metrics, not full PPO training curves.
   - Current workaround: `single_task/train_ppo.py` now writes `eval_metrics.csv` and `eval_f1_curve.png` during training.
   - Impact: to fully match the PDF, we still need either recover original logs or run a full training job.

10. Shared-filesystem stalls can make Python startup/checkpoint loading unreliable.
   - Symptom: one standalone PyTorch import and the `Numb_1` high-level eval spent time in `rpc_wait_bit_killable`.
   - Impact: long-running experiments may appear stuck while waiting on NFS/shared storage.
   - Mitigation to consider: run from local scratch storage or copy checkpoints/datasets to a local disk before launching many jobs.
   - Current status: runtime repo/data/checkpoints have been copied to `/home/gaoj/piano_scratch/pianomime`; the GPU watcher runs from this local copy.

11. Codebase maintainability issues remain.
   - Many scripts use CWD-dependent paths such as `dataset/...` and `handtracking/...`.
   - Several modules mutate `sys.path` instead of using package-relative imports.
   - Shell scripts hard-code CUDA device choices and assume a particular launch directory.
   - Some dataset/model modules print debug output from library code.
   - Current status: the most important eval/replay/training entrypoints and shell scripts were cleaned up; see `/home/gaoj/share4/_piano/code_audit.md`.

## Fixed Or Partially Fixed In Code

- Multi-task eval scripts now have CPU/GPU device fallback.
- Multi-task eval scripts now resolve artifact paths from the repo root.
- `robopianist.music.midi_file` no longer requires `pyaudio` on import.
- `robopianist.wrappers.sound` can render silent videos when FluidSynth is unavailable.
- `single_task/train_ppo.py` now creates checkpoint/action output directories before saving.
- `single_task/train_ppo.py` now saves PPO evaluation metrics and a F1 curve image.

## 2026-05-14 Additional Notes

12. GitHub publishing still needs a remote repository.
   - The project directory now has `.gitignore` and release instructions.
   - The server does not have `gh` installed, and the available GitHub connector cannot create a new empty repository.
   - NFS-backed shared storage cannot write Git loose objects reliably here, so the repo uses worktree `/home/gaoj/share4/_piano/pianomime` with local gitdir `/home/gaoj/piano_scratch/pianomime_gitdir`.
   - Local commit exists; check the exact current hash with `git log --oneline -1`.
   - Next step: create an empty GitHub repository manually, then push from `/home/gaoj/share4/_piano/pianomime`.

13. RTX 4090 feasibility has not been physically verified on this server.
   - No 4090 is currently visible here through the checked tools.
   - Based on checkpoint/model size and PyTorch CUDA requirements, one baseline job per 24 GB 4090 should be feasible.
   - A smoke script was added: `scripts/check_4090_feasibility.sh`.
   - Do not report this as a completed hardware verification until it has actually run on a 4090 host.

14. The old GPU watcher was started outside tmux.
   - It is useful but does not satisfy the requirement that jobs be inspectable through tmux.
   - The replacement tmux scheduler has been added and should be used going forward.

15. A lazy-import cleanup initially missed wrappers used by `multi_task/get_env_ll`.
   - Symptom: `Numb_1` and `NoTimeToDie_1` low-level eval failed with `NameError: name 'DeepMimicWrapper' is not defined`.
   - Cause: `multi_task/utils.py` moved environment dependencies into `_env_deps()`, but did not include `DeepMimicWrapper`, `FingeringEmbWrapper`, and `ResidualWrapper`.
   - Fix: added those wrappers to `_env_deps()` and unpacked them in the relevant env-construction functions.
   - Status: fixed in shared and scratch copies; failed low-level evals were restarted.

16. One accidental duplicate PPO launch was immediately stopped.
   - Cause: the first tmux low-level rerun used empty `PPO_TASKS=''`, but the scheduler treated empty as missing and used the default `Petrunko_3`.
   - Fix: scheduler defaults now distinguish unset from intentionally empty task lists.
   - Status: duplicate `Petrunko_3_ppo_curve_lowlevel_fix_20260514` process was killed; original `Petrunko_3_ppo_curve_20260513_135059` continues.

17. Shared/NFS dependency reads can prevent jobs from reaching CUDA allocation.
   - Symptom: four high-level eval processes for the A800 batch are in `wait_on_page_bit_common` or `rpc_wait_bit_killable`.
   - Impact: `nvidia-smi` may show idle GPUs even though Python jobs were launched.
   - Mitigation used: compiled a local CUDA guard program to reserve memory directly, without Python imports.
   - Current status: the A800 batch eventually left the NFS wait, finished successfully, and the guards were released.
   - Longer-term fix if this recurs: build a fully local Python runtime/env under `/home/gaoj/piano_scratch` instead of using the shared `.venv` plus `/home/gaoj/anaconda3`.
