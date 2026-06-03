# PianoMime Baseline 结果

最后更新：2026-06-03

本文档是 Dexterous Piano Track baseline 复现结果的主要索引。后续同学开始做改进之前，建议先读这里，确认 baseline 数字、视频、日志和训练曲线分别在哪里。

## 结果目录

所有复现出来的 metrics、logs、videos 和 training traces 都放在：

```text
/home/gaoj/share4/_piano/baseline_results
```

重要文件结构：

```text
baseline_results/
  single_song/metrics.csv
  single_song/videos/
  single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/
  single_song/training_runs/Stan_1_ppo_curve_20260603_110918/  # running
  multisong/metrics.csv
  multisong/videos/
  multisong/logs/
```

当前视频是 silent video，因为服务器缺少系统级 FluidSynth/PortAudio 库。视觉仿真、MIDI evaluation 和 note-level metrics 仍然有效。

## PDF Baseline 要求对照

| PDF 中的 baseline 要求 | 当前状态 |
| --- | --- |
| 在原始 PianoMime repo 上做实验 | 已完成；只做了服务器运行所需的工程修复。 |
| 从 training dataset 选 3 首歌 | 已完成：`Stan_1`、`Petrunko_3`、`NeverGonnaGiveYouUp_1`。 |
| 产出 final performance videos | 已完成；三个 single-song replay videos 见下文。 |
| 可视化 F1 score training curve | 已完成；`Petrunko_3` PPO residual curve 已保存。 |
| 训练 multi-song policy 或使用 repo 给定 checkpoint | 已完成；使用 released high-level 和 low-level diffusion checkpoints。 |
| 在 unseen/test songs 上展示 performance video | 已完成；7 首 test songs 都有 metrics 和 videos。 |

## Single-Song Replay Baseline

PDF 原始要求是从 training dataset 选 3 首并产出 final performance videos；这部分已经由旧三首 replay baseline 完成。后续同学的 single-song 改进统一改为四首集合：`TwinkleTwinkleRousseau`、`Pirates_1`、`Stan_1`、`Petrunko_3`。四首集合的完整状态见 `docs/SINGLE_SONG_FOUR_BASELINE_zh.md`。

下面是目前已经有同口径 action replay artifacts 的曲目结果。它们使用 provided single-song low-level action trajectories，并保持原始 PianoMime/RoboPianist evaluation wrappers。

| Song | Precision | Recall | F1 | Duration | Video |
| --- | ---: | ---: | ---: | ---: | --- |
| `Stan_1` | 0.9991 | 0.9719 | 0.9795 | 26.90 s | `/home/gaoj/share4/_piano/baseline_results/single_song/videos/Stan_1_single_song_baseline.mp4` |
| `Petrunko_3` | 0.9869 | 0.8460 | 0.8900 | 30.05 s | `/home/gaoj/share4/_piano/baseline_results/single_song/videos/Petrunko_3_single_song_baseline.mp4` |
| `NeverGonnaGiveYouUp_1` | 0.9960 | 0.9260 | 0.9514 | 28.00 s | `/home/gaoj/share4/_piano/baseline_results/single_song/videos/NeverGonnaGiveYouUp_1_single_song_baseline.mp4` |

四首对齐集合状态：

| Song | Action replay | PPO residual baseline |
| --- | --- | --- |
| `TwinkleTwinkleRousseau` | 原始 release 缺少 low-level actions，不能 replay | smoke test 因 MIDI/demo 长度不一致失败 |
| `Pirates_1` | 原始 release 缺少 low-level actions，不能 replay | smoke test 因 IK/QP prior 初始化失败 |
| `Stan_1` | 已完成，F1 0.9795 | 正在补跑：`Stan_1_ppo_curve_20260603_110918` |
| `Petrunko_3` | 已完成，F1 0.8900 | 已完成，best F1 0.795686 |

原始表格：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/metrics.csv
```

5 秒处预览图：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/video_previews/Stan_1_single_song_baseline_t5s.png
/home/gaoj/share4/_piano/baseline_results/single_song/video_previews/Petrunko_3_single_song_baseline_t5s.png
/home/gaoj/share4/_piano/baseline_results/single_song/video_previews/NeverGonnaGiveYouUp_1_single_song_baseline_t5s.png
```

日志：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/logs/Petrunko_3.log
/home/gaoj/share4/_piano/baseline_results/single_song/logs/NeverGonnaGiveYouUp_1.log
```

`Stan_1` 是在 stdout 重定向规范化之前跑的，所以它的指标记录在 CSV 和视频文件中，但没有单独的 log 文件。

## Single-Song PPO Training Curve

PDF 要求提供 F1 training curve。Released checkpoints 不包含 training logs，所以我们用原始 baseline 超参重新跑了 `Petrunko_3` PPO residual training，并额外保存了曲线和 CSV。

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

注意：这条 PPO run 属于 residual single-song 路径。它的 prior action 来自 stored song demonstration trajectory 经过 IK/QP 的结果，不来自 generalist diffusion checkpoint。`02001.mp4` 是 best checkpoint rollout，因此对应 best recorded F1。

## Generalist Diffusion Baseline

Generalist baseline 使用 released high-level 和 low-level diffusion checkpoints。High-level diffusion 生成 fingertip trajectories，low-level diffusion 根据预测 trajectory 和 simulator observations 生成可执行 robot actions。

| Song | Split | Precision | Recall | F1 | Video |
| --- | --- | ---: | ---: | ---: | --- |
| `Alone_1` | test | 0.8283 | 0.6443 | 0.7902 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Alone_1_multisong_baseline.mp4` |
| `Numb_1` | test | 0.5286 | 0.4521 | 0.7504 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Numb_1_multisong_baseline.mp4` |
| `NoTimeToDie_1` | test | 0.8192 | 0.8076 | 0.8553 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/NoTimeToDie_1_multisong_baseline.mp4` |
| `Forester_1` | test | 0.8116 | 0.7300 | 0.7944 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Forester_1_multisong_baseline.mp4` |
| `EyesClosed_1` | test | 0.6127 | 0.5151 | 0.8569 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/EyesClosed_1_multisong_baseline.mp4` |
| `Paradise_1` | test | 0.8392 | 0.7535 | 0.8104 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/Paradise_1_multisong_baseline.mp4` |
| `SomewhereOnlyWeKnow_1` | test | 0.6516 | 0.5789 | 0.7920 | `/home/gaoj/share4/_piano/baseline_results/multisong/videos/SomewhereOnlyWeKnow_1_multisong_baseline.mp4` |

原始表格：

```text
/home/gaoj/share4/_piano/baseline_results/multisong/metrics.csv
```

每首歌的日志：

```text
/home/gaoj/share4/_piano/baseline_results/multisong/logs/<song>_high_level.log
/home/gaoj/share4/_piano/baseline_results/multisong/logs/<song>_low_level.log
```

## Baseline 完成状态

课程 proposal/milestone 所需的 baseline 部分已经完成：

- 3 首 training-set single-song videos 和 F1 scores 已有。
- 一条 single-song PPO F1 training curve 已有。
- 7 首 unseen-song generalist checkpoint videos 和 F1 scores 已有。
- 日志和 metrics 都保存在共享结果目录，方便后续比较。

剩余工作属于 improvement phase：

- 提升 single-song F1，并和当前 baseline 对比。
- 提升 generalist F1，并至少在 5 首 pieces 上对比。
- 如果展示必须有声音，可以安装系统音频库并重新渲染视频。
