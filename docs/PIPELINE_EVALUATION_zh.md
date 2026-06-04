# Baseline Pipeline 说明

最后更新：2026-05-19

本文档面向后续继续维护 PianoMime 项目的同学，说明当前 baseline pipeline 已经支持哪些功能、如何使用，以及扩展新方法时需要遵守哪些约定。

## 功能范围

| 能力 | 当前状态 | 主要入口 |
| --- | --- | --- |
| 环境搭建 | 已提供课程服务器上的 Python 环境脚本 | `scripts/setup_python_env.sh` |
| 数据和 checkpoint 准备 | 可复用缓存或从官方链接下载 artifacts | `scripts/setup_artifacts.sh` |
| Single-song replay baseline | 已支持生成 F1、日志和视频 | `scripts/test_trained_actions.sh` |
| Single-song PPO residual training | 已支持保存 eval CSV、F1 曲线、checkpoint 和 rollout video | `scripts/run_ppo.sh` |
| Generalist diffusion baseline | 已支持 high-level + low-level checkpoint evaluation | `scripts/run_multisong_task.sh` |
| 长实验运行 | 已支持 tmux 后台执行 | `scripts/start_tmux_baseline.sh` |
| GPU 等待 | scheduler 会按显存阈值等待 GPU 空闲后启动任务 | `scripts/baseline_scheduler.sh` |
| 共享盘 I/O 缓解 | 默认在本地 scratch 目录运行，再同步结果到共享目录 | `scripts/sync_to_runtime.sh` |
| 结果归档 | metrics、logs、videos 有统一目录和索引文档 | `docs/BASELINE_RESULTS_zh.md` |
| 集中配置 | baseline 路径、任务列表和核心超参已集中到 TOML | `configs/baseline.toml` |

## 推荐使用流程

### 1. 准备环境和 artifacts

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_python_env.sh
bash scripts/setup_artifacts.sh
```

默认配置会创建或复用：

```text
/home/gaoj/share4/_piano/.venv
/home/gaoj/share4/_piano/artifacts
/home/gaoj/share4/_piano/baseline_results
/home/gaoj/piano_scratch/pianomime
/home/gaoj/piano_scratch/baseline_results
```

如果在其他机器或目录运行，优先修改 `configs/baseline.toml` 中的 `[paths]`，也可以通过环境变量覆盖脚本导出的默认值。

### 2. 运行单个 baseline

Single-song replay：

```bash
bash scripts/test_trained_actions.sh Stan_1
```

Single-song PPO residual training：

```bash
bash scripts/run_ppo.sh Petrunko_3
```

Generalist diffusion evaluation：

```bash
bash scripts/run_multisong_task.sh Alone_1 0
```

### 3. 使用 tmux 批量运行

```bash
cd /home/gaoj/share4/_piano/pianomime
GPU_IDS="4 5 6 7" bash scripts/start_tmux_baseline.sh
```

查看后台任务：

```bash
tmux attach -t pianomime_baseline
```

查看日志：

```bash
tail -f /home/gaoj/share4/_piano/baseline_results/logs/scheduler_*.log
tail -f /home/gaoj/share4/_piano/baseline_results/logs/tmux_*.log
```

## 集中配置

核心配置文件是：

```text
configs/baseline.toml
```

常用字段包括：

| 配置段 | 内容 |
| --- | --- |
| `[paths]` | repo、venv、artifacts、共享结果目录、本地 scratch 目录 |
| `[artifacts]` | 官方 dataset/checkpoint 的 Google Drive id 和缓存文件名 |
| `[environment]` | `MUJOCO_GL`、JAX preallocation、W&B 模式 |
| `[scheduler]` | tmux session、GPU 显存阈值、轮询间隔、runtime sync |
| `[single_song]` | replay/PPO 默认曲目列表 |
| `[single_song.replay]` | single-song replay 环境和视频参数 |
| `[single_song.ppo]` | PPO residual training 的核心超参 |
| `[multisong]` | generalist baseline 曲目列表 |
| `[multisong.high_level]` | high-level diffusion eval 参数 |
| `[multisong.low_level]` | low-level diffusion eval 参数 |

脚本会默认读取 `configs/baseline.toml`。如需使用另一份配置：

```bash
CONFIG_FILE=/path/to/your_config.toml bash scripts/start_tmux_baseline.sh
```

建议新方法不要直接改 baseline 配置，而是复制一份，例如：

```bash
cp configs/baseline.toml configs/my_method.toml
CONFIG_FILE=configs/my_method.toml bash scripts/run_multisong_task.sh Alone_1 0
```

这样可以保留 baseline 可复现性，也方便后续对比。

## 已整理的 baseline 数字

Single-song replay：

| Song | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `Stan_1` | 0.9991 | 0.9719 | 0.9795 |
| `Petrunko_3` | 0.9869 | 0.8460 | 0.8900 |
| `NeverGonnaGiveYouUp_1` | 0.9960 | 0.9260 | 0.9514 |

后续 single-song 改进的统一集合为 `TwinkleTwinkleRousseau`、`Pirates_1`、`Stan_1`、`Petrunko_3`。当前 pipeline 已经能把四首写入配置并做 artifacts preflight；`TwinkleTwinkleRousseau`/`Pirates_1` 的 demo/MIDI 对齐和 IK/QP 数值问题已修复，完整 2000-iteration PPO baseline 已经跑完，指标和文件路径见 `docs/SINGLE_SONG_FOUR_BASELINE_zh.md`。

Single-song PPO residual：

| Song | Iterations | Best-checkpoint rollout F1 |
| --- | ---: | ---: |
| `Petrunko_3` | 2000 | 0.795686 |
| `TwinkleTwinkleRousseau` | 2000 | 0.7912 |
| `Pirates_1` | 2000 | 0.8718 |

Generalist diffusion：

| Song | F1 |
| --- | ---: |
| `Alone_1` | 0.7902 |
| `Numb_1` | 0.7504 |
| `NoTimeToDie_1` | 0.8553 |
| `Forester_1` | 0.7944 |
| `EyesClosed_1` | 0.8569 |
| `Paradise_1` | 0.8104 |
| `SomewhereOnlyWeKnow_1` | 0.7920 |

完整路径、视频和日志索引见 `docs/BASELINE_RESULTS_zh.md`。

## 维护约定

后续同学实现新方法时，建议遵守以下约定：

1. 不要覆盖 `baseline_results/` 中已有 baseline 结果。
2. 新实验应使用新的 config 文件或新的结果子目录。
3. 每次实验至少记录 `method`、`song`、`seed`、`checkpoint`、`git_commit`、`command`。
4. 保留 stdout/stderr log、metrics CSV、关键视频和曲线图。
5. 修改 baseline 入口脚本时，同步更新 `configs/baseline.toml` 和使用文档。

当前代码已经覆盖课程 baseline&pipeline 分工所需的主要流程：环境准备、artifacts 准备、single-song/generalist baseline 运行、PPO 曲线保存、tmux 长实验、GPU 等待、本地 scratch 运行、结果同步和文档索引。后续算法改进可以直接基于这套流程进行对比。
