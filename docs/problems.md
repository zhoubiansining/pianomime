# PianoMime Problems Log

Last updated: 2026-06-03

This document separates resolved engineering problems from caveats that remain
important for future teammates.

## Resolved or Mitigated

1. Missing dataset and checkpoints.
   - Fixed by restoring the official PianoMime artifacts.
   - Future setup: `bash scripts/setup_artifacts.sh`.

2. Missing or incompatible Python dependencies.
   - Fixed for this server with `/home/gaoj/share4/_piano/.venv`.
   - `requirements.txt` now records the tested package pins.
   - CUDA PyTorch is installed by `scripts/setup_python_env.sh`.

3. CPU-only PyTorch in the original environment.
   - Fixed in the project virtualenv with CUDA 11.8 PyTorch wheels.

4. CWD-dependent scripts.
   - The main eval/replay/training entrypoints now resolve paths from
     `PROJECT_ROOT`.

5. Multi-task eval assuming `.cuda()`.
   - Fixed with device-aware checkpoint loading and encoder execution.

6. Unnecessary autograd graph construction during eval observation encoding.
   - Fixed with `torch.no_grad()` in representative encoder calls.

7. Audio dependency import failures.
   - Mitigated by allowing silent video output when FluidSynth/PortAudio system
     libraries are unavailable.

8. Missing PPO training-curve logs.
   - Fixed by adding `eval_metrics.csv` and `eval_f1_curve.png` output to
     `single_task/train_ppo.py`.

9. Shared-filesystem stalls during long experiments.
   - Mitigated by running from `/home/gaoj/piano_scratch/pianomime` and syncing
     final results back to `/home/gaoj/share4/_piano/baseline_results`.

10. Lack of experiment persistence after disconnect.
    - Mitigated with tmux automation and rerunnable scheduler logic.

## Remaining Caveats

1. Videos are silent.
   - Cause: missing system FluidSynth/PortAudio libraries.
   - Impact: videos are visually valid but have no audio track.
   - Fix only if needed for presentation: install system audio packages and
     rerender selected videos.

2. 4090 feasibility is not physically verified on this host.
   - The model/checkpoint sizes should fit one job per 24 GB 4090.
   - Use `scripts/check_4090_feasibility.sh` on a real 4090 machine before
     claiming verified 4090 support.

3. Fresh GitHub clone has not been tested yet.
   - This is blocked on the new empty remote repository URL.
   - After push, clone into a new directory and run:
     `scripts/setup_artifacts.sh`, `scripts/check_4090_feasibility.sh`, and one
     short eval command.

4. System-level dependencies are still partly outside `requirements.txt`.
   - MuJoCo EGL, ffmpeg, and optional audio libraries depend on the server.
   - `docs/USAGE.md` documents the expected environment variables and setup.

5. Algorithmic improvement work has not started.
   - Baseline reproduction is done.
   - The next phase is improving single-song and generalist F1, not more
     baseline cleanup.

6. Two songs in the four-song single-song alignment set do not run under the
   original residual baseline artifacts yet.
   - Aligned set: `TwinkleTwinkleRousseau`, `Pirates_1`, `Stan_1`,
     `Petrunko_3`.
   - `TwinkleTwinkleRousseau`: the available fingertip demo trajectory has 150
     frames, while the built-in MIDI task expands to 451 note steps, so
     `DeepMimicWrapper` fails its length assertion.
   - `Pirates_1`: notes and high-level trajectories exist, but residual-prior
     initialization hits an infeasible IK/QP solve and triggers
     `assert dq is not None`.
   - See `docs/SINGLE_SONG_FOUR_BASELINE.md` for smoke-test logs.

## Do Not Forget

- Do not report the single-song PPO prior as coming from the diffusion
  generalist. It comes from stored demonstration trajectories plus IK/QP.
- Do not change reward terms, task definitions, or baseline hyperparameters
  when running baseline comparisons.
- Keep new experiment outputs traceable with CSV metrics, logs, and videos.
