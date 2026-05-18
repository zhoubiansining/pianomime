# Baseline & Pipeline 能力评估

最后更新：2026-05-18

本文档回答两个问题：

1. 当前仓库是否已经足够承担 baseline&pipeline 这部分工作？
2. 如果把它称为 pipeline，它已经实现了哪些功能，还有哪些不足？

## 简短结论

当前代码库已经可以作为课程项目的 **baseline reproduction pipeline** 使用，也可以作为后续同学做改进实验的 **基础实验 pipeline**。

更精确地说：

- 对于“复现 baseline、保存指标、保存视频、保存日志、给后续改进提供对比基准”这个目标：已经足够。
- 对于“像成熟研究平台一样支持任意新方法、统一配置管理、自动汇总所有实验、完整失败重试、跨机器无痛部署”这个目标：还没有完全达到。

因此公允评价是：**当前是一个可用且可维护的课程 baseline pipeline，不是一个完全工业化/平台化的实验系统。**

## Pipeline 应该实现的功能

面向我们负责的 baseline&pipeline 部分，一个合格 pipeline 应该覆盖以下能力。

| 能力 | 期望效果 | 当前是否实现 | 评价 |
| --- | --- | --- | --- |
| 环境搭建 | 同学能按文档创建可运行 Python/CUDA 环境 | 基本实现 | `scripts/setup_python_env.sh` 和 `requirements.txt` 已有，但系统级 MuJoCo/EGL/audio 仍依赖服务器。 |
| 数据和 checkpoint 准备 | 能恢复 PianoMime 官方 dataset/checkpoints | 已实现 | `scripts/setup_artifacts.sh` 会复用缓存或下载 artifacts。 |
| Single-song replay baseline | 能对 training songs 产出 F1 和视频 | 已实现 | 3 首歌结果已完成，并保存在 `baseline_results`。 |
| Single-song PPO training curve | 能训练 residual PPO 并保存 F1 curve | 已实现 | `Petrunko_3` 已跑 2000 iterations，代码会保存 CSV 和曲线。 |
| Generalist diffusion baseline | 能跑 high-level + low-level diffusion checkpoint eval | 已实现 | 7 首 unseen songs 已有结果。 |
| 结果归档 | metrics、logs、videos 可追溯 | 已实现 | `baseline_results` 结构清晰，另有 `BASELINE_RESULTS_zh.md` 索引。 |
| 长实验抗断开 | 退出 VSCode/Codex 后实验继续跑 | 已实现 | 使用 tmux runner。 |
| GPU 等待和领取 | GPU 被占用时等待，空闲后启动任务 | 部分实现 | Scheduler 会按显存阈值等待和 lock；不是集群级调度系统。 |
| 共享盘 I/O 缓解 | 避免大量仿真/视频写共享目录 | 已实现 | 通过 `/home/gaoj/piano_scratch` runtime copy 执行，再同步结果。 |
| 重复运行/恢复 | 已完成任务跳过，中断任务可重跑 | 部分实现 | CSV/video 已存在会跳过；PPO 有 checkpoint resume，但不是 step-level 精准恢复。 |
| 文档交接 | 同学能理解 single-song/generalist 区别和如何跑 | 已实现 | 中英文文档已补齐。 |
| 新方法扩展 | 同学能在 baseline 上实现改进并对比 | 基本实现 | 有清晰入口和 baseline 数字，但还缺统一 experiment config schema。 |
| 自动汇总报告 | 自动生成最终表格/图表/报告 | 部分实现 | baseline 表格已整理，未来新实验还需手动或补脚本聚合。 |

## 当前 Pipeline 的实际流程

### 1. 环境和 artifacts

目标：让同学可以恢复一个能运行的 PianoMime 环境。

入口：

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_python_env.sh
bash scripts/setup_artifacts.sh
```

效果：

- 创建或复用 `/home/gaoj/share4/_piano/.venv`。
- 安装 CUDA PyTorch 和当前测试过的依赖。
- 恢复 dataset、`dataset_hl.zarr`、`dataset_ll.zarr`、`checkpoint_high_level.ckpt`、`checkpoint_low_level.ckpt`、`checkpoint_ae.ckpt`。

评价：作为课程服务器 pipeline 是够用的。fresh machine 上仍可能需要同学处理系统级 EGL/ffmpeg/audio 库。

### 2. Single-song baseline

目标：复现 single-song policy 侧 baseline，并保存视频和指标。

入口：

```bash
bash scripts/test_trained_actions.sh Stan_1
bash scripts/run_ppo.sh Petrunko_3
```

或使用自动 scheduler：

```bash
GPU_IDS="4 5 6 7" bash scripts/start_tmux_baseline.sh
```

已完成结果：

- `Stan_1` replay F1: 0.9795
- `Petrunko_3` replay F1: 0.8900
- `NeverGonnaGiveYouUp_1` replay F1: 0.9514
- `Petrunko_3` PPO best/final F1: 0.795686

评价：single-song baseline pipeline 已经满足 PDF baseline 要求。后续如果要比较 improved PPO，只需要沿用同样 metrics/video 输出格式。

### 3. Generalist diffusion baseline

目标：复现 high-level diffusion + low-level diffusion 的 unseen-song generalist baseline。

入口：

```bash
python multi_task/eval_high_level.py Alone_1
python multi_task/eval_low_level.py Alone_1
```

或使用：

```bash
bash scripts/run_multisong_task.sh Alone_1 0
```

已完成结果：

- `Alone_1`: F1 0.7902
- `Numb_1`: F1 0.7504
- `NoTimeToDie_1`: F1 0.8553
- `Forester_1`: F1 0.7944
- `EyesClosed_1`: F1 0.8569
- `Paradise_1`: F1 0.8104
- `SomewhereOnlyWeKnow_1`: F1 0.7920

评价：generalist baseline pipeline 已经足够支持后续改进比较。课程要求后续 generalist improvement 要在 5 首曲子上比较，当前 baseline 已经准备了 7 首。

### 4. 结果管理

目标：所有同学都能找到结果，不需要翻日志猜测。

当前结构：

```text
/home/gaoj/share4/_piano/baseline_results/
  single_song/metrics.csv
  single_song/videos/
  single_song/video_previews/
  single_song/training_runs/
  multisong/metrics.csv
  multisong/videos/
  multisong/logs/
```

索引文档：

```text
docs/BASELINE_RESULTS_zh.md
docs/BASELINE_RESULTS.md
```

评价：baseline 阶段足够清晰。后续 improvement 阶段建议新增 `improvement_results/` 或在结果 CSV 中增加 `method`、`run_id`、`seed`、`notes` 等字段。

## 当前使用说明是否达到 pipeline 程度

结论：**基本达到了 baseline pipeline 的使用程度。**

原因：

- 文档说明了如何准备环境、准备 artifacts、跑 single-song、跑 generalist、跑 PPO、用 tmux 自动跑长实验。
- 已完成结果有统一索引，不需要同学重新跑 baseline 才知道 baseline 数字。
- 代码修改总结明确告诉同学哪些文件被动过，哪些逻辑不能随便改。
- 问题文档列出了 silent video、4090 未物理验证、fresh GitHub clone 尚待验证等真实 caveats。

不足也要坦诚：

- 目前 pipeline 仍以 shell scripts 和约定目录为主，不是 Hydra/Sacred/W&B sweep 那种统一 experiment management。
- Config 还不够集中，很多 baseline hyperparameters 仍分散在脚本参数和 Python 默认值里。
- 结果聚合是“已有 baseline 已整理好”，但未来新方法的大批量结果还没有自动生成最终报告的脚本。
- 失败恢复是 job-level，不是精确到每个 env step 或 diffusion step 的恢复。
- 跨服务器部署仍依赖共享目录约定和部分系统库，不能保证任意机器 one-command 成功。

## 是否可以交给同学使用

可以，但建议明确告诉同学它的定位：

```text
这是课程项目的 baseline reproduction pipeline。
它适合用来：
1. 查看和复现当前 baseline；
2. 作为 single-song 和 generalist 改进实验的起点；
3. 统一保存 metrics、logs 和 videos；
4. 用 tmux 在服务器上稳定跑长实验。

它还不是完整实验平台。
做新方法时，请继续记录配置、结果和问题。
```

## 建议后续补强

如果时间允许，pipeline 还能继续增强：

1. 增加统一实验配置文件，例如 `configs/baseline.yaml`、`configs/improvement.yaml`。
2. 增加 `scripts/collect_results.py`，自动汇总 baseline/improvement metrics 并生成 Markdown/CSV 表。
3. 增加 `improvement_results/` 目录规范，避免后续新实验覆盖 baseline。
4. 对每次实验记录 `method`、`song`、`seed`、`checkpoint`、`git_commit`、`command`。
5. 在 GitHub remote 建好后，做一次真正的 fresh-clone artifact setup 和短 eval。

## 最终评价

以课程 baseline&pipeline 分工来看，当前完成度可以评价为：

| 维度 | 评价 |
| --- | --- |
| Baseline 复现 | 完成 |
| Baseline 结果整理 | 完成 |
| 同学阅读文档 | 基本完成，中英文都有 |
| 长实验运行 | 可用 |
| 新方法对比基础 | 可用 |
| 完整实验平台化 | 尚未完成 |

因此，如果汇报时需要一句话总结，可以说：

```text
我们已经完成了 PianoMime baseline 的复现和课程项目所需的实验 pipeline：它可以自动准备环境和 artifacts，运行 single-song 与 generalist baseline，保存 F1、视频、日志和训练曲线，并支持 tmux 长实验和共享目录结果同步。后续算法改进可以直接基于这个 pipeline 进行对比。
```
