# PianoMime Baseline 计划和状态

最后更新：2026-05-19

Dexterous Piano Track PDF 要求的 baseline reproduction 已经完成。

## 课程要求摘要

该 track 要求我们：

- 回顾 robot piano playing 相关工作，并介绍 RoboPianist、PianoMime、RP1M。
- 运行原始 PianoMime baseline。
- 从 training dataset 选择 3 首适合 robot 演奏的歌。
- 产出 final performance videos。
- 可视化 F1 score training curve。
- 训练 multi-song policy 或使用 provided checkpoints。
- 在 unseen/test songs 上展示 generalist policy performance videos。
- 后续提升 single-song policy 和 generalist policy。

## Baseline 复现状态

PDF baseline 要求已经完成。

已完成事项：

- 环境和依赖已安装在 `/home/gaoj/share4/_piano/.venv`。
- 官方 PianoMime datasets 和 checkpoints 已恢复到项目目录。
- RoboPianist 可以在 headless MuJoCo EGL 模式下运行。
- 已评估 3 首 training-set single-song replay baselines：`Stan_1`、`Petrunko_3`、`NeverGonnaGiveYouUp_1`。
- `Petrunko_3` PPO residual training 已跑 2000 iterations，并产出 `eval_metrics.csv`、`eval_f1_curve.png` 和 final rollout video。
- 已完成 7 首 unseen-song generalist diffusion checkpoint evaluations：`Alone_1`、`Numb_1`、`NoTimeToDie_1`、`Forester_1`、`EyesClosed_1`、`Paradise_1`、`SomewhereOnlyWeKnow_1`。
- 结果文档集中在 `docs/BASELINE_RESULTS_zh.md`。
- 代码修改文档集中在 `docs/CODE_MODIFICATION_SUMMARY_zh.md`。
- 路径、任务列表、scheduler 默认值和核心 baseline 超参数已集中到 `configs/baseline.toml`，配置说明见 `docs/CONFIGURATION_zh.md`。

## 结果位置

共享结果目录：

```text
/home/gaoj/share4/_piano/baseline_results
```

主要结果文件：

```text
single_song/metrics.csv
single_song/videos/
single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/
multisong/metrics.csv
multisong/videos/
multisong/logs/
```

## 当前代码库状态

这个仓库已经可以给同学查看和用于后续实验，但需要注意：

- 当前 worktree 是 Git repository，后续 push 到 GitHub 前应保持 clean。
- 大型 artifacts 被 `.gitignore` 忽略，并通过 `scripts/setup_artifacts.sh` 恢复。
- 复现 baseline 时使用 `configs/baseline.toml`；后续新方法建议复制新配置再改，避免污染 baseline。
- 长实验应使用 `scripts/start_tmux_baseline.sh` 和 local scratch execution，减少共享文件系统卡顿。
- 当前服务器上的 A800 run 已验证。4090 有 smoke script，但本机未看到 4090，因此尚未物理验证。

## 下一阶段研究工作

剩下的不是 baseline 复现，而是算法改进：

- 选择一个 single-song improvement target，与记录好的 baseline 对比 F1 curve 和 play-through video。
- 选择一个 generalist improvement idea，在至少 5 首 unseen songs 上对比 baseline 和 improved F1。
- 新实验建议沿用 `baseline_results` 的组织方式，或者新增平行的 `improvement_results` 目录。

## 可选维护工作

- 用户创建新的空 GitHub remote 后，把当前本地 Git repo push 上去。
- remote 存在后做一次 fresh-clone smoke test。
- 只有展示需要声音时，才安装 FluidSynth/PortAudio 系统库。
