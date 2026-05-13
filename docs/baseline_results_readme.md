# PianoMime Baseline Results

Updated: 2026-05-12

This directory collects reproducible outputs for the Dexterous Piano Track baseline.

## Environment

Use the local environment and repo path:

```bash
source /home/gaoj/share4/_piano/.venv/bin/activate
export MUJOCO_GL=egl
export PYTHONPATH=/home/gaoj/share4/_piano/pianomime
cd /home/gaoj/share4/_piano/pianomime
```

Current caveats:

- The project venv now has `torch==2.1.0+cu118`, so CUDA is available when GPUs are free.
- Videos are silent because system PortAudio/FluidSynth libraries are unavailable.
- The repo directory is not currently a Git checkout, so changes cannot yet be audited with `git diff`.
- Long-running GPU jobs are scheduled by `/home/gaoj/share4/_piano/run_gpu_baselines.sh`; current scheduler log is `/home/gaoj/piano_scratch/baseline_results/gpu_scheduler_20260513_135059.log`.

## Single-Song Baseline

See `single_song/metrics.csv`.

Interpretation: these are demonstration-conditioned single-song policy results. The prior action comes from the song's external fingertip demo trajectory through IK/QP, and PPO learns a residual correction. This path is separate from the generalist diffusion policy.

Videos:

- `single_song/videos/Stan_1_single_song_baseline.mp4`
- `single_song/videos/Petrunko_3_single_song_baseline.mp4`
- `single_song/videos/NeverGonnaGiveYouUp_1_single_song_baseline.mp4`

These are action-replay results from the provided low-level policies, not newly trained PPO curves.

## Multi-Song Baseline

High-level checkpoint evaluation has completed for unseen test song `Alone_1`.

Interpretation: this is the generalist diffusion baseline. The high-level diffusion policy generates trajectories from MIDI/goal, and the low-level diffusion policy generates robot actions. This path is separate from single-song PPO.

Low-level checkpoint evaluation has also completed:

- `multisong/logs/Alone_1_low_level.log`
- `multisong/videos/Alone_1_multisong_baseline.mp4`
- `multisong/metrics.csv`

`Alone_1` metrics: precision `0.8283`, recall `0.6443`, F1 `0.7902`.

An additional attempt on `Numb_1` was started but stopped during high-level startup because the process stalled in shared-filesystem I/O. Its partial log is kept at `multisong/logs/Numb_1_high_level.log`.

The high-level intermediate outputs live in:

- `/home/gaoj/share4/_piano/pianomime/multi_task/trajectories/Alone_1_trajectory.npy`
- `/home/gaoj/share4/_piano/pianomime/multi_task/trajectories/Alone_1_left_hand_action_list.npy`
- `/home/gaoj/share4/_piano/pianomime/multi_task/trajectories/Alone_1_right_hand_action_list.npy`
