# Course Baseline Notes

Last updated: 2026-05-19

This fork is the course working copy for the Dexterous Piano Track. It keeps
the original PianoMime baseline logic intact, while adding reproducibility
scripts, headless-server fixes, baseline result logging, and maintenance notes.

## Start Here

- Chinese overview: `COURSE_BASELINE_zh.md`
- Full reproduced result index: `docs/BASELINE_RESULTS.md`
- Chinese result index: `docs/BASELINE_RESULTS_zh.md`
- Report-ready baseline LaTeX snippet: `docs/BASELINE_REPORT_SECTION.tex`
- Chinese report materials guide: `docs/BASELINE_REPORT_MATERIALS_zh.md`
- Chinese report asset manifest: `docs/REPORT_ASSETS_MANIFEST_zh.md`
- Detailed code-change summary: `docs/CODE_MODIFICATION_SUMMARY.md`
- Chinese code-change summary: `docs/CODE_MODIFICATION_SUMMARY_zh.md`
- Configuration guide: `docs/CONFIGURATION.md`
- Chinese configuration guide: `docs/CONFIGURATION_zh.md`
- Usage and setup guide: `docs/USAGE.md`
- Chinese usage guide: `docs/USAGE_zh.md`
- Unattended tmux runner: `docs/EXPERIMENT_AUTOMATION.md`
- Chinese tmux runner guide: `docs/EXPERIMENT_AUTOMATION_zh.md`
- Current caveats: `docs/problems.md`
- Chinese caveats: `docs/problems_zh.md`
- Chinese pipeline evaluation: `docs/PIPELINE_EVALUATION_zh.md`

## Code Paths

Single-song policy:

```text
single_task/train_ppo.py
single_task/test_trained_actions.py
single_task/utils.py
```

This path uses a fixed song demonstration trajectory plus an IK/QP prior.
PPO learns a residual correction on top of that prior action. It does not load
the multi-task high-level or low-level diffusion checkpoints.

Generalist policy:

```text
multi_task/eval_high_level.py
multi_task/eval_low_level.py
multi_task/utils.py
```

This path uses the released diffusion checkpoints. The high-level model
generates fingertip trajectories from MIDI/goal observations, and the low-level
model generates executable robot actions from those trajectories and simulator
observations. It does not use single-song PPO checkpoints.

## Current Baseline Results

All result files are under:

```text
/home/gaoj/share4/_piano/baseline_results
```

Single-song replay baseline:

| Song | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `Stan_1` | 0.9991 | 0.9719 | 0.9795 |
| `Petrunko_3` | 0.9869 | 0.8460 | 0.8900 |
| `NeverGonnaGiveYouUp_1` | 0.9960 | 0.9260 | 0.9514 |

Downstream single-song improvements are aligned on four songs:
`TwinkleTwinkleRousseau`, `Pirates_1`, `Stan_1`, and `Petrunko_3`. `Stan_1`
and `Petrunko_3` have same-protocol action replay results. The first two songs
now pass residual PPO smoke tests after demo/MIDI alignment and IK/QP fallback
fixes; full 2000-iteration runs are in progress. See
`docs/SINGLE_SONG_FOUR_BASELINE.md`.

Single-song PPO curve:

| Song | Iterations | Env steps | Best F1 | Output |
| --- | ---: | ---: | ---: | --- |
| `Petrunko_3` | 2000 | 1,024,000 | 0.795686 | `eval_metrics.csv`, `eval_f1_curve.png`, final rollout video |

Generalist diffusion checkpoint baseline:

| Song | Split | F1 |
| --- | --- | ---: |
| `Alone_1` | test | 0.7902 |
| `Numb_1` | test | 0.7504 |
| `NoTimeToDie_1` | test | 0.8553 |
| `Forester_1` | test | 0.7944 |
| `EyesClosed_1` | test | 0.8569 |
| `Paradise_1` | test | 0.8104 |
| `SomewhereOnlyWeKnow_1` | test | 0.7920 |

For exact paths to videos, logs, and CSV files, see
`docs/BASELINE_RESULTS.md`.

## Baseline Status

The baseline reproduction required by the project PDF is complete:

- 3 training-set single-song videos and metrics are available.
- A PPO F1 training curve is available.
- 7 unseen-song generalist videos and metrics are available.
- Experiments leave logs and reusable CSV files for later comparison.
- Paths, task lists, and core baseline hyperparameters are centralized in
  `configs/baseline.toml`; copy it for improvement experiments.

The next research step is no longer baseline reproduction; it is implementing
and evaluating improvement ideas against these recorded baseline numbers.

## Important Commands

Prepare artifacts and environment:

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_python_env.sh
bash scripts/setup_artifacts.sh
```

Start or reattach the automated tmux baseline runner:

```bash
cd /home/gaoj/share4/_piano/pianomime
GPU_IDS="4 5 6 7" SESSION=pianomime_baseline RUN_ID=baseline_$(date +%Y%m%d) \
  bash scripts/start_tmux_baseline.sh
tmux attach -t pianomime_baseline
```

Detach without stopping experiments:

```text
Ctrl-b then d
```

Inspect result files:

```bash
cat /home/gaoj/share4/_piano/baseline_results/single_song/metrics.csv
cat /home/gaoj/share4/_piano/baseline_results/multisong/metrics.csv
```
