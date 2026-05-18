# 实验自动化说明

自动 baseline runner 是：

```bash
scripts/start_tmux_baseline.sh
```

它会启动一个 detached tmux session，并在其中运行：

```bash
scripts/baseline_scheduler.sh
```

## 启动

当前共享服务器上的典型启动方式：

```bash
cd /home/gaoj/share4/_piano/pianomime
GPU_IDS="5 6 7" SESSION=pianomime_baseline RUN_ID=baseline_20260514 \
  bash scripts/start_tmux_baseline.sh
```

在其他服务器上，选择当前空闲的 GPU id：

```bash
GPU_IDS="0 1 2" bash scripts/start_tmux_baseline.sh
```

如果省略 `GPU_IDS`，scheduler 会查询所有可见 GPU，并等待某张卡的显存占用低于 `GPU_FREE_MEM_MB`。默认阈值是 `5000` MiB。

## 进入、退出和日志

进入 tmux：

```bash
tmux attach -t pianomime_baseline
```

退出 tmux 但不停止实验：

```text
Ctrl-b 然后按 d
```

主要日志：

```bash
tail -f /home/gaoj/share4/_piano/baseline_results/logs/tmux_baseline_20260514.log
tail -f /home/gaoj/share4/_piano/baseline_results/logs/scheduler_baseline_20260514.log
```

任务日志：

```text
baseline_results/single_song/logs/
baseline_results/single_song/training_runs/
baseline_results/multisong/logs/
```

视频：

```text
baseline_results/single_song/videos/
baseline_results/multisong/videos/
```

Metrics：

```text
baseline_results/single_song/metrics.csv
baseline_results/multisong/metrics.csv
```

## Resume 行为

Scheduler 设计为可以重复运行：

- 如果 single-song replay 已经有 metrics row 和 video，会跳过。
- 如果 generalist eval 已经有 metrics row 和 video，会跳过。
- 如果 high-level trajectories 已经生成，会在 low-level eval 中复用。
- PPO 使用稳定的 `RUN_ID`；如果本地 run directory 中已有 best checkpoint，下一次运行会通过 `--pretrained` 传入。
- PPO 完成后会在共享 training output directory 中创建 `.done`。

这提供的是 job-level recovery。如果进程在一次 iteration 中间被杀，可以用相同 `RUN_ID` 重新执行同一个命令。

## 默认任务集合

Scheduler 默认运行：

```text
SINGLE_REPLAY_TASKS="Stan_1 Petrunko_3 NeverGonnaGiveYouUp_1"
MULTISONG_TASKS="Alone_1 Numb_1 NoTimeToDie_1"
PPO_TASKS="Petrunko_3"
```

当前已经完成的 baseline 还额外包括 4 首 unseen generalist songs：

```text
Forester_1 EyesClosed_1 Paradise_1 SomewhereOnlyWeKnow_1
```

可以通过环境变量覆盖任务集合：

```bash
MULTISONG_TASKS="Numb_1 NoTimeToDie_1" PPO_TASKS="Petrunko_3" \
  GPU_IDS="5 6 7" bash scripts/start_tmux_baseline.sh
```

## I/O 策略

项目源码和最终结果放在共享目录。每次运行开始时，会先把项目同步到 `/home/gaoj/piano_scratch/pianomime`，然后从本地副本执行实验。这样可以降低共享文件系统上的仿真和视频写入压力。
