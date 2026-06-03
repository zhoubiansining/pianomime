# Four-Song Single-Song Baseline Alignment

Last updated: 2026-06-03

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
| `TwinkleTwinkleRousseau` | no released low-level actions | smoke passed; full run: `TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548` |
| `Pirates_1` | no released low-level actions | smoke passed; full run: `Pirates_1_ppo_curve_fix2_20260603_114548` |
| `Stan_1` | available, F1 0.9795 | interrupted old run; rerunnable with the same config |
| `Petrunko_3` | available, F1 0.8900 | available, best F1 0.795686 |

Smoke-test metrics:

| Song | Env steps | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| `TwinkleTwinkleRousseau` | 512 | 0.1876 | 0.6959 | 0.4196 |
| `Pirates_1` | 512 | 0.8697 | 0.8036 | 0.8304 |

The smoke runs only validate the pipeline fixes. Full 2000-iteration PPO
baselines are running in:

```bash
tmux attach -t pianomime_twinkle_pirates_fix
tail -f /home/gaoj/piano_scratch/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548.log
tail -f /home/gaoj/piano_scratch/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548.log
```

Fix summary:

- `TwinkleTwinkleRousseau`: set per-song `control_timestep = 0.15` and pad the
  150-frame demo trajectory to the 151-step MIDI task.
- `Pirates_1`: add IK/QP solver fallback after `quadprog` numerical failure.
- `single_task/utils.py`: align demonstration length to task length and pass the
  current `control_timestep` as `demo_ctrl_timestep`.

