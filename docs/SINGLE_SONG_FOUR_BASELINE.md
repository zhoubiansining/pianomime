# Four-Song Single-Song Baseline Alignment

Last updated: 2026-06-03

The downstream single-song improvement experiments are aligned on these four songs:

```text
TwinkleTwinkleRousseau
Pirates_1
Stan_1
Petrunko_3
```

They are now recorded in `configs/baseline.toml` under `single_song.baseline_songs` and `single_song.ppo_songs`. The original PianoMime release does not provide complete action-replay artifacts for all four songs, so the current status is split as follows.

`TwinkleTwinkleRousseau` and `Pirates_1` are also listed in
`single_song.ppo_blocked_songs` so the default scheduler does not repeatedly
launch known-failing jobs. Clear `PPO_BLOCKED_TASKS=""` or update the config
after matching artifacts or an IK/QP fix is available.

| Song | Action replay video | Action replay F1 | PPO residual curve |
| --- | --- | ---: | --- |
| `Stan_1` | available | 0.9795 | running: `Stan_1_ppo_curve_20260603_110918` |
| `Petrunko_3` | available | 0.8900 | available: best F1 0.795686 |
| `TwinkleTwinkleRousseau` | no released low-level actions | unavailable | smoke test failed |
| `Pirates_1` | no released low-level actions | unavailable | smoke test failed |

`TwinkleTwinkleRousseau` fails the residual single-song smoke test because the built-in MIDI task expands to 451 note steps while the available fingertip demonstration trajectory has 150 frames. `Pirates_1` has notes and fingertip trajectories, but the residual prior initialization hits an infeasible IK/QP solve.

The smoke-test logs are:

```text
/home/gaoj/piano_scratch/baseline_results/single_song/smoke_runs/TwinkleTwinkleRousseau_smoke_20260603.log
/home/gaoj/piano_scratch/baseline_results/single_song/smoke_runs/Pirates_1_smoke_20260603.log
```

`Stan_1` PPO is running in:

```bash
tmux attach -t pianomime_single_song_four
tail -f /home/gaoj/piano_scratch/baseline_results/single_song/training_runs/Stan_1_ppo_curve_20260603_110918.log
```

Code/config changes are limited to the central config, per-song PPO overrides
in `scripts/run_ppo_from_config.py`, replay artifact preflight in
`scripts/baseline_scheduler.sh`, a clearer artifact check in
`single_task/test_trained_actions.py`, and `pianomime_config.py` handling of
explicitly empty task-list environment variables.
