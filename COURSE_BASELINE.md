# Course Baseline Notes

This fork is the course working copy for the Dexterous Piano Track. It keeps the
original PianoMime task logic intact, while adding reproducibility scripts,
headless-server fixes, baseline result logging, and maintenance notes.

## Code Paths

- Single-song policy: `single_task/train_ppo.py`
  - Uses a fixed song demonstration trajectory plus IK prior.
  - PPO learns a residual correction on top of that prior action.
  - It does not load the multi-task diffusion checkpoints.
- Single-song action replay baseline: `single_task/test_trained_actions.py`
  - Replays the provided action trajectory for a song and records precision,
    recall, F1, and a video.
- Generalist policy: `multi_task/eval_high_level.py` and
  `multi_task/eval_low_level.py`
  - High-level diffusion generates hand trajectories from the MIDI/goal.
  - Low-level diffusion generates executable actions from those trajectories.
  - This is the code path used for unseen-song generalization.

The single-song and generalist policies are conceptually related but are
separate implementation paths.

## Current Baseline Results

Results are synced to:

```bash
/home/gaoj/share4/_piano/baseline_results
```

Already reproduced:

| Track | Song | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| single-song replay | `Stan_1` | 0.9991 | 0.9719 | 0.9795 |
| single-song replay | `Petrunko_3` | 0.9869 | 0.8460 | 0.8900 |
| single-song replay | `NeverGonnaGiveYouUp_1` | 0.9960 | 0.9260 | 0.9514 |
| generalist diffusion | `Alone_1` | 0.8283 | 0.6443 | 0.7902 |

Still scheduled/ongoing:

- Generalist diffusion: `Numb_1`, `NoTimeToDie_1`
- Single-song PPO residual training curve: `Petrunko_3`

## Important Commands

Start or reattach the automated tmux baseline runner:

```bash
cd /home/gaoj/share4/_piano/pianomime
GPU_IDS="5 6 7" SESSION=pianomime_baseline RUN_ID=baseline_20260514 \
  bash scripts/start_tmux_baseline.sh
tmux attach -t pianomime_baseline
```

Detach without stopping experiments:

```text
Ctrl-b then d
```

Inspect logs:

```bash
tail -f /home/gaoj/share4/_piano/baseline_results/logs/tmux_baseline_20260514.log
tail -f /home/gaoj/share4/_piano/baseline_results/logs/scheduler_baseline_20260514.log
```

See also:

- `docs/USAGE.md`
- `docs/EXPERIMENT_AUTOMATION.md`
- `docs/4090_FEASIBILITY.md`
- `docs/CODEX_HANDOFF_PROMPT.md`
- `docs/code_audit.md`
- `docs/problems.md`
- `docs/planning.md`
