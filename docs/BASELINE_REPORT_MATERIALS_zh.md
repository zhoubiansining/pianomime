# Baseline 报告展示材料

最后更新：2026-05-26

本文档用于准备课程报告、poster 或 milestone slides 中的 baseline 部分。它不是复现实验说明，而是把“报告里应该展示什么”和“对应文件在哪里”集中列出，方便同学直接取表格、图片和视频。

## PDF 要求对照

| 课程文档要求 | 当前准备状态 | 报告中建议展示 |
| --- | --- | --- |
| 在原始 PianoMime repo 上做实验 | 已完成；只做了路径、无显示服务器、日志和配置化等工程修复 | 简短说明基于原始 PianoMime baseline，未改变算法定义 |
| 从 training dataset 选 3 首歌 | 已完成：`Stan_1`、`Petrunko_3`、`NeverGonnaGiveYouUp_1` | 表格列出三首歌的 Precision / Recall / F1 |
| 产出 final performance videos | 已完成 | slide 或报告附录列出 3 个视频路径，展示时选 1-3 个短片段 |
| 可视化 F1 score training curve | 已完成：`Petrunko_3` PPO residual training | 插入 `eval_f1_curve.png`，说明训练 2000 iterations |
| 训练 multi-song policy 或使用给定 checkpoints | 已完成；使用 released high-level / low-level diffusion checkpoints | 说明使用官方 checkpoint 作为 generalist baseline |
| 在 unseen/test songs 上展示 performance video | 已完成 7 首 test songs | 表格列出 7 首 F1；展示时选 5 首满足后续 improvement 对比要求 |

## 可直接放进报告的 Baseline 表格

### Single-song replay baseline

| Song | Split | Precision | Recall | F1 | Duration |
| --- | --- | ---: | ---: | ---: | ---: |
| `Stan_1` | train | 0.9991 | 0.9719 | 0.9795 | 26.90 s |
| `Petrunko_3` | train | 0.9869 | 0.8460 | 0.8900 | 30.05 s |
| `NeverGonnaGiveYouUp_1` | train | 0.9960 | 0.9260 | 0.9514 | 28.00 s |

建议正文表述：

```text
For the single-song baseline, we replayed the provided song-specific demonstration trajectories with the original PianoMime/RoboPianist evaluation wrappers. The three selected training-set songs cover different note densities and durations, and all produced valid play-through videos and note-level F1 scores.
```

### Single-song PPO residual training curve

| Item | Value |
| --- | --- |
| Song | `Petrunko_3` |
| Iterations | 2000 |
| Environment steps | 1,024,000 |
| Best recorded F1 | 0.795686 |
| Final rollout F1 | 0.795686 |

建议正文表述：

```text
The single-song residual PPO baseline was rerun on Petrunko_3 to recover the F1 training curve required by the project documentation. Following the original residual-policy setting, PPO learns a correction on top of an IK/QP prior derived from the provided demonstration trajectory.
```

### Generalist diffusion checkpoint baseline

| Song | Split | Precision | Recall | F1 | Duration |
| --- | --- | ---: | ---: | ---: | ---: |
| `Alone_1` | test | 0.8283 | 0.6443 | 0.7902 | 30.05 s |
| `Numb_1` | test | 0.5286 | 0.4521 | 0.7504 | 30.05 s |
| `NoTimeToDie_1` | test | 0.8192 | 0.8076 | 0.8553 | 30.05 s |
| `Forester_1` | test | 0.8116 | 0.7300 | 0.7944 | 27.20 s |
| `EyesClosed_1` | test | 0.6127 | 0.5151 | 0.8569 | 30.05 s |
| `Paradise_1` | test | 0.8392 | 0.7535 | 0.8104 | 30.05 s |
| `SomewhereOnlyWeKnow_1` | test | 0.6516 | 0.5789 | 0.7920 | 30.05 s |

建议正文表述：

```text
For the generalist baseline, we evaluated the released two-stage diffusion policy on unseen test songs. The high-level diffusion model predicts fingertip trajectories from MIDI/goal observations, and the low-level diffusion model converts the trajectories and simulator observations into executable robot actions.
```

## 报告图片和视频路径

### PPO F1 curve

```text
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/eval_f1_curve.png
```

### Single-song videos

```text
/home/gaoj/share4/_piano/baseline_results/single_song/videos/Stan_1_single_song_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/single_song/videos/Petrunko_3_single_song_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/single_song/videos/NeverGonnaGiveYouUp_1_single_song_baseline.mp4
```

### PPO final rollout video

```text
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/eval/02001.mp4
```

### Generalist videos

```text
/home/gaoj/share4/_piano/baseline_results/multisong/videos/Alone_1_multisong_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/multisong/videos/Numb_1_multisong_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/multisong/videos/NoTimeToDie_1_multisong_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/multisong/videos/Forester_1_multisong_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/multisong/videos/EyesClosed_1_multisong_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/multisong/videos/Paradise_1_multisong_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/multisong/videos/SomewhereOnlyWeKnow_1_multisong_baseline.mp4
```

## 报告中需要说明的 caveats

- 当前视频是 silent video。原因是服务器缺少系统级 FluidSynth/PortAudio 库；视觉仿真和 note-level metrics 不受影响。
- Single-song PPO 的 prior action 来自外部 demonstration trajectory 经过 IK/QP 的结果，不来自 generalist diffusion policy。
- Generalist baseline 使用 released checkpoints，没有重新训练 high-level 或 low-level diffusion policy。
- GitHub 仓库当前包含 `dataset_hl.zarr` 和 `dataset_ll.zarr`，方便复现 generalist baseline；如果后续仓库体积成为问题，可以改回 artifact download/cache 方式。

## 建议在报告中的组织方式

1. 在 method 或 experiment setup 中先区分 single-song residual PPO 与 generalist diffusion policy。
2. 在 baseline subsection 中放一张 requirement mapping 小表，说明 PDF 要求均已覆盖。
3. 放 single-song replay 表格和 1-2 个视频截图或视频链接。
4. 放 PPO F1 curve，并说明该曲线用于满足 baseline training-curve 要求。
5. 放 generalist 7 首 test songs 的指标表；后续 improvement 至少选其中 5 首做对比。
6. 在 limitations 中说明 silent video 和 checkpoint-eval 设置。
