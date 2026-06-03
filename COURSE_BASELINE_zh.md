# 课程 Baseline 说明

最后更新：2026-05-19

这是 Dexterous Piano Track 的课程工作版本。当前仓库尽量保持原始 PianoMime baseline 的算法逻辑不变，同时补充了可复现脚本、无显示服务器上的运行修复、baseline 结果记录和维护说明。

## 建议先读

- 完整 baseline 结果索引：`docs/BASELINE_RESULTS_zh.md`
- 报告/展示材料清单：`docs/BASELINE_REPORT_MATERIALS_zh.md`
- 可直接粘贴到报告的 LaTeX 小节：`docs/BASELINE_REPORT_SECTION.tex`
- 报告图片资产清单：`docs/REPORT_ASSETS_MANIFEST_zh.md`
- 后续工作提纲：`docs/NEXT_STEPS_zh.md`
- 代码修改详细说明：`docs/CODE_MODIFICATION_SUMMARY_zh.md`
- 使用和环境配置说明：`docs/USAGE_zh.md`
- 集中配置说明：`docs/CONFIGURATION_zh.md`
- tmux 自动实验说明：`docs/EXPERIMENT_AUTOMATION_zh.md`
- 当前问题和注意事项：`docs/problems_zh.md`
- pipeline 能力评估：`docs/PIPELINE_EVALUATION_zh.md`

## 代码路径

Single-song policy：

```text
single_task/train_ppo.py
single_task/test_trained_actions.py
single_task/utils.py
```

这一路径使用某一首歌固定的 demonstration trajectory，并通过 IK/QP 生成 prior action。PPO 学习的是叠加在 prior action 上的 residual correction。它不加载 multi-task high-level 或 low-level diffusion checkpoints。

Generalist policy：

```text
multi_task/eval_high_level.py
multi_task/eval_low_level.py
multi_task/utils.py
```

这一路径使用 released diffusion checkpoints。High-level model 根据 MIDI/goal observation 生成 fingertip trajectories，low-level model 根据这些 trajectories 和仿真 observation 生成可执行的 robot actions。它不使用 single-song PPO checkpoints。

## 当前 Baseline 结果

所有结果文件都放在：

```text
/home/gaoj/share4/_piano/baseline_results
```

Single-song replay baseline：

| Song | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `Stan_1` | 0.9991 | 0.9719 | 0.9795 |
| `Petrunko_3` | 0.9869 | 0.8460 | 0.8900 |
| `NeverGonnaGiveYouUp_1` | 0.9960 | 0.9260 | 0.9514 |

后续 single-song 改进统一对齐到四首：`TwinkleTwinkleRousseau`、`Pirates_1`、`Stan_1`、`Petrunko_3`。其中 `Stan_1` 和 `Petrunko_3` 已有同口径 action replay 结果；`TwinkleTwinkleRousseau` 和 `Pirates_1` 的 residual PPO smoke test 已跑通，正式 2000-iteration baseline 正在运行，详情见 `docs/SINGLE_SONG_FOUR_BASELINE_zh.md`。

Single-song PPO curve：

| Song | Iterations | Env steps | Best F1 | Output |
| --- | ---: | ---: | ---: | --- |
| `Petrunko_3` | 2000 | 1,024,000 | 0.795686 | `eval_metrics.csv`、`eval_f1_curve.png`、final rollout video |

Generalist diffusion checkpoint baseline：

| Song | Split | F1 |
| --- | --- | ---: |
| `Alone_1` | test | 0.7902 |
| `Numb_1` | test | 0.7504 |
| `NoTimeToDie_1` | test | 0.8553 |
| `Forester_1` | test | 0.7944 |
| `EyesClosed_1` | test | 0.8569 |
| `Paradise_1` | test | 0.8104 |
| `SomewhereOnlyWeKnow_1` | test | 0.7920 |

视频、日志和 CSV 的精确路径见 `docs/BASELINE_RESULTS_zh.md`。

## Baseline 状态

课程 PDF 中 baseline 部分要求的复现已经完成：

- 已有 3 首 training-set single-song 的视频和指标。
- 已有一条 PPO F1 training curve。
- 四首 single-song 对齐集合已写入配置；`TwinkleTwinkleRousseau`/`Pirates_1` 的数据对齐和 IK/QP 问题已修复，正式 PPO baseline 正在 tmux 中运行。
- 已有 7 首 unseen-song generalist 的视频和指标。
- 实验留下了日志和可复用的 CSV 文件，便于后续和改进方法对比。
- 路径、曲目列表和核心 baseline 超参数已集中到 `configs/baseline.toml`，同学可以复制新配置来做改进实验。

下一步研究工作不再是 baseline 复现，而是基于这些 baseline 数字实现和评估改进方法。

## 常用命令

准备环境和数据：

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_python_env.sh
bash scripts/setup_artifacts.sh
```

启动或进入 tmux 自动 baseline runner：

```bash
cd /home/gaoj/share4/_piano/pianomime
GPU_IDS="4 5 6 7" SESSION=pianomime_baseline RUN_ID=baseline_$(date +%Y%m%d) \
  bash scripts/start_tmux_baseline.sh
tmux attach -t pianomime_baseline
```

从 tmux 里退出但不停止实验：

```text
Ctrl-b 然后按 d
```

查看结果：

```bash
cat /home/gaoj/share4/_piano/baseline_results/single_song/metrics.csv
cat /home/gaoj/share4/_piano/baseline_results/multisong/metrics.csv
```
