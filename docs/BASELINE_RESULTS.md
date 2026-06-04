# PianoMime Baseline Results

Last updated: 2026-06-04

This document is the main index for the reproduced Dexterous Piano Track
baseline. It records the exact result files that classmates should inspect
before starting improvement work.

## Result Directory

All reproduced metrics, logs, videos, and training traces are stored under:

```text
/home/gaoj/share4/_piano/baseline_results
```

Important files:

```text
baseline_results/
  single_song/metrics.csv
  single_song/videos/
  single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/
  single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548/
  single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548/
  multisong/metrics.csv
  multisong/videos/
  multisong/logs/
```

The videos are currently silent because the server is missing system
FluidSynth/PortAudio libraries. The visual simulation, MIDI evaluation, and
note-level metrics are still valid.

## PDF Baseline Requirements

| Requirement from project PDF | Current status |
| --- | --- |
| Run experiments on the original PianoMime repo | Done, with engineering-only fixes for server execution. |
| Select 3 songs from the training dataset | Done: `Stan_1`, `Petrunko_3`, `NeverGonnaGiveYouUp_1`. |
| Produce final performance videos | Done: three single-song replay videos are listed below. |
| Visualize the F1 score training curve | Done: PPO residual curve for `Petrunko_3`. |
| Train a multi-song policy or use provided checkpoints | Done: used released high-level and low-level diffusion checkpoints. |
| Show performance video on unseen/test songs | Done: seven test songs have metrics and videos. |

## Single-Song Replay Baseline

The original PDF requirement asks for three training-set songs with final
performance videos; that part is complete with the replay baseline below.
Downstream single-song improvements are now aligned on four songs:
`TwinkleTwinkleRousseau`, `Pirates_1`, `Stan_1`, and `Petrunko_3`. See
`docs/SINGLE_SONG_FOUR_BASELINE.md` for the full four-song status.

The table below lists songs for which same-protocol action-replay artifacts are
available. They replay the provided single-song low-level action trajectories
with the original PianoMime/RoboPianist evaluation wrappers.

| Song | Precision | Recall | F1 | Duration | Video |
| --- | ---: | ---: | ---: | ---: | --- |
| `Stan_1` | 0.9991 | 0.9719 | 0.9795 | 26.90 s | `/home/gaoj/share4/_piano/baseline_results/single_song/videos/Stan_1_single_song_baseline.mp4` |
| `Petrunko_3` | 0.9869 | 0.8460 | 0.8900 | 30.05 s | `/home/gaoj/share4/_piano/baseline_results/single_song/videos/Petrunko_3_single_song_baseline.mp4` |
| `NeverGonnaGiveYouUp_1` | 0.9960 | 0.9260 | 0.9514 | 28.00 s | `/home/gaoj/share4/_piano/baseline_results/single_song/videos/NeverGonnaGiveYouUp_1_single_song_baseline.mp4` |

Four-song alignment status:

| Song | Action replay | PPO residual baseline |
| --- | --- | --- |
| `TwinkleTwinkleRousseau` | missing released low-level actions | complete, best-checkpoint rollout F1 0.7912 |
| `Pirates_1` | missing released low-level actions | complete, best-checkpoint rollout F1 0.8718 |
| `Stan_1` | done, F1 0.9795 | rerunnable with the same config |
| `Petrunko_3` | done, F1 0.8900 | done, best F1 0.795686 |

Raw table:

```text
/home/gaoj/share4/_piano/baseline_results/single_song/metrics.csv
```

Preview frames extracted at 5 seconds:

```text
/home/gaoj/share4/_piano/baseline_results/single_song/video_previews/Stan_1_single_song_baseline_t5s.png
/home/gaoj/share4/_piano/baseline_results/single_song/video_previews/Petrunko_3_single_song_baseline_t5s.png
/home/gaoj/share4/_piano/baseline_results/single_song/video_previews/NeverGonnaGiveYouUp_1_single_song_baseline_t5s.png
```

Logs:

```text
/home/gaoj/share4/_piano/baseline_results/single_song/logs/Petrunko_3.log
/home/gaoj/share4/_piano/baseline_results/single_song/logs/NeverGonnaGiveYouUp_1.log
```

`Stan_1` was run before stdout redirection was standardized, so its metric is
recorded in the CSV and video file but has no separate log file.

## Single-Song PPO Training Curve

The PDF asks for an F1 training curve. The released checkpoints did not include
training logs, so `Petrunko_3` PPO residual training was rerun with the original
baseline hyperparameters and extra logging.

| Item | Value |
| --- | --- |
| Song | `Petrunko_3` |
| Run name | `Petrunko_3_ppo_curve_20260513_135059` |
| Iterations | 2000 |
| Environment steps | 1,024,000 |
| Best recorded F1 | 0.795686 |
| Best-checkpoint rollout F1 | 0.795686 |
| Last evaluation F1 | 0.686684 |
| Curve image | `/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/eval_f1_curve.png` |
| Metrics CSV | `/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/eval_metrics.csv` |
| Final rollout video | `/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/eval/02001.mp4` |
| Training log | `/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059.log` |

Interpretation note: this PPO run is the residual single-song path. Its prior
action comes from the stored song demonstration trajectory through IK/QP, not
from the generalist diffusion checkpoint. `02001.mp4` is generated from the
best checkpoint and therefore corresponds to the best recorded F1.

Additional four-song-alignment PPO baselines:

| Song | Iterations | Env steps | Best-checkpoint rollout P/R/F1 | Last evaluation P/R/F1 | Run directory |
| --- | ---: | ---: | --- | --- | --- |
| `TwinkleTwinkleRousseau` | 2000 | 1,024,000 | 0.7693 / 0.7025 / 0.7912 | 0.9338 / 0.6639 / 0.6406 | `/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548` |
| `Pirates_1` | 2000 | 1,024,000 | 0.9064 / 0.8722 / 0.8718 | 0.8369 / 0.7950 / 0.8177 | `/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548` |

## Generalist Diffusion Baseline

The generalist baseline uses the released high-level and low-level diffusion
checkpoints. High-level diffusion generates fingertip trajectories, and
low-level diffusion generates executable robot actions from the predicted
trajectory and simulator observations.

| Song | Split | Precision | Recall | F1 | Video |
| --- | --- | ---: | ---: | ---: | --- |
| `Alone_1` | test | 0.8283 | 0.6443 | 0.7902 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Alone_1_multisong_baseline.mp4` |
| `Numb_1` | test | 0.5286 | 0.4521 | 0.7504 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Numb_1_multisong_baseline.mp4` |
| `NoTimeToDie_1` | test | 0.8192 | 0.8076 | 0.8553 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/NoTimeToDie_1_multisong_baseline.mp4` |
| `Forester_1` | test | 0.8116 | 0.7300 | 0.7944 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Forester_1_multisong_baseline.mp4` |
| `EyesClosed_1` | test | 0.6127 | 0.5151 | 0.8569 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/EyesClosed_1_multisong_baseline.mp4` |
| `Paradise_1` | test | 0.8392 | 0.7535 | 0.8104 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Paradise_1_multisong_baseline.mp4` |
| `SomewhereOnlyWeKnow_1` | test | 0.6516 | 0.5789 | 0.7920 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/SomewhereOnlyWeKnow_1_multisong_baseline.mp4` |

Raw table:

```text
/home/gaoj/share4/_piano/baseline_results/multisong/metrics.csv
```

Per-song logs:

```text
/home/gaoj/share4/_piano/baseline_results/multisong/logs/<song>_high_level.log
/home/gaoj/share4/_piano/baseline_results/multisong/logs/<song>_low_level.log
```

## Baseline Completion Status

The baseline portion is complete for the course proposal/milestone purpose:

- Three training-set single-song videos and F1 scores are available.
- A single-song PPO F1 training curve is available.
- Seven unseen-song generalist checkpoint videos and F1 scores are available.
- Logs and metrics are kept in a shared result directory.

Remaining work belongs to the improvement phase:

- Improve single-song F1 and compare against this baseline.
- Improve generalist F1 and compare on at least five pieces.
- Optionally install system audio libraries if sound is required in videos.
