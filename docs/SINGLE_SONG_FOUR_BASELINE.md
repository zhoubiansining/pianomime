# Four-Song Single-Song Baseline Alignment

Last updated: 2026-06-04

Downstream single-song improvement experiments are aligned on:

```text
TwinkleTwinkleRousseau
Pirates_1
Stan_1
Petrunko_3
```

These songs are recorded in `configs/baseline.toml` under
`single_song.baseline_songs` and `single_song.ppo_songs`.

| Song | Action replay baseline | PPO residual baseline |
| --- | --- | --- |
| `TwinkleTwinkleRousseau` | no released low-level actions | complete, best-checkpoint rollout F1 0.7912 |
| `Pirates_1` | no released low-level actions | complete, best-checkpoint rollout F1 0.8718 |
| `Stan_1` | available, F1 0.9795 | interrupted old run; rerunnable with the same config |
| `Petrunko_3` | available, F1 0.8900 | available, best F1 0.795686 |

Full 2000-iteration PPO metrics:

| Song | Iterations | Env steps | Best-checkpoint rollout P/R/F1 | Last evaluation P/R/F1 | Curve / video |
| --- | ---: | ---: | --- | --- | --- |
| `TwinkleTwinkleRousseau` | 2000 | 1,024,000 | 0.7693 / 0.7025 / 0.7912 | 0.9338 / 0.6639 / 0.6406 | `TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548` |
| `Pirates_1` | 2000 | 1,024,000 | 0.9064 / 0.8722 / 0.8718 | 0.8369 / 0.7950 / 0.8177 | `Pirates_1_ppo_curve_fix2_20260603_114548` |

Result files:

```text
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548/eval_metrics.csv
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548/eval_f1_curve.png
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548/eval/02001.mp4

/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548/eval_metrics.csv
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548/eval_f1_curve.png
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548/eval/02001.mp4
```

Fix summary:

- `TwinkleTwinkleRousseau`: set per-song `control_timestep = 0.15` and pad the
  150-frame demo trajectory to the 151-step MIDI task.
- `Pirates_1`: add IK/QP solver fallback after `quadprog` numerical failure.
- `single_task/utils.py`: align demonstration length to task length and pass the
  current `control_timestep` as `demo_ctrl_timestep`.
