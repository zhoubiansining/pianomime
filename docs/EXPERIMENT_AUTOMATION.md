# Experiment Automation

The automated baseline runner is:

```bash
scripts/start_tmux_baseline.sh
```

It starts a detached tmux session that runs:

```bash
scripts/baseline_scheduler.sh
```

## Start

On the current shared server setup:

```bash
cd /home/gaoj/share4/_piano/pianomime
GPU_IDS="5 6 7" SESSION=pianomime_baseline RUN_ID=baseline_20260514 \
  bash scripts/start_tmux_baseline.sh
```

On another server, choose the visible idle GPU ids:

```bash
GPU_IDS="0 1 2" bash scripts/start_tmux_baseline.sh
```

If `GPU_IDS` is omitted, the scheduler queries all visible GPUs and waits until
one is below `GPU_FREE_MEM_MB`, which defaults to `5000`.

## Attach, Detach, Logs

Attach:

```bash
tmux attach -t pianomime_baseline
```

Detach:

```text
Ctrl-b then d
```

Primary logs:

```bash
tail -f /home/gaoj/share4/_piano/baseline_results/logs/tmux_baseline_20260514.log
tail -f /home/gaoj/share4/_piano/baseline_results/logs/scheduler_baseline_20260514.log
```

Task logs:

```text
baseline_results/single_song/logs/
baseline_results/single_song/training_runs/
baseline_results/multisong/logs/
```

Videos:

```text
baseline_results/single_song/videos/
baseline_results/multisong/videos/
```

Metrics:

```text
baseline_results/single_song/metrics.csv
baseline_results/multisong/metrics.csv
```

## Resume Behavior

The scheduler is designed to be rerunnable:

- Existing single-song replay rows plus videos are skipped.
- Existing generalist rows plus videos are skipped.
- Existing high-level generated trajectories are reused for low-level eval.
- PPO uses a stable `RUN_ID`; if a best checkpoint exists in the local run
  directory, the next run passes it through `--pretrained`.
- PPO completion creates `.done` inside the shared training output directory.

This gives job-level recovery. If the process is killed mid-iteration, rerun the
same command with the same `RUN_ID`.

## Default Task Set

The scheduler defaults to:

```text
SINGLE_REPLAY_TASKS="Stan_1 Petrunko_3 NeverGonnaGiveYouUp_1"
MULTISONG_TASKS="Alone_1 Numb_1 NoTimeToDie_1"
PPO_TASKS="Petrunko_3"
```

The completed baseline result suite also includes four additional unseen
generalist songs:

```text
Forester_1 EyesClosed_1 Paradise_1 SomewhereOnlyWeKnow_1
```

Override any set through environment variables:

```bash
MULTISONG_TASKS="Numb_1 NoTimeToDie_1" PPO_TASKS="Petrunko_3" \
  GPU_IDS="5 6 7" bash scripts/start_tmux_baseline.sh
```

## I/O Policy

The project source and final results live in the shared directory. Each run
starts by syncing the project into `/home/gaoj/piano_scratch/pianomime`, then
executes from that local copy. This avoids known slowdowns from doing all
simulation/video writes directly on the shared filesystem.
