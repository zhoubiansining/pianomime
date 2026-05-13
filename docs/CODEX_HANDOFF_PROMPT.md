# Codex Handoff Prompt

Copy the following prompt into a new Codex session after the GitHub repository
has been created.

```text
你现在负责继续维护和复现一个 Deep Reinforcement Learning 课程项目：Dexterous Piano Track / PianoMime baseline。

共享目录约定：
- 所有服务器都能访问 `/home/gaoj/share4`。
- 共享项目根目录建议使用 `/home/gaoj/share4/_piano`。
- 源码仓库应放在 `/home/gaoj/share4/_piano/pianomime`。
- 最终日志、metrics、视频和训练痕迹必须同步到 `/home/gaoj/share4/_piano/baseline_results`。
- 运行时不要直接在共享盘大量写仿真视频和中间文件；先把项目同步到本机 scratch，例如 `/home/gaoj/piano_scratch/pianomime`，实验从 scratch 跑，结果再 rsync 回共享目录。

如果本地还没有仓库：
1. `mkdir -p /home/gaoj/share4/_piano`
2. `cd /home/gaoj/share4/_piano`
3. `git clone <这里填新建 GitHub 仓库 URL> pianomime`

请先阅读：
- `COURSE_BASELINE.md`
- `docs/USAGE.md`
- `docs/EXPERIMENT_AUTOMATION.md`
- `docs/4090_FEASIBILITY.md`
- `docs/planning.md`
- `docs/problems.md`
- `docs/code_audit.md`

任务背景：
- single-song policy 与 generalist diffusion policy 是分开的代码路径。
- single-song PPO 在 `single_task/train_ppo.py`，使用某首歌已有 demonstration trajectory 和 IK prior，PPO 只学习 residual correction。
- single-song action replay baseline 在 `single_task/test_trained_actions.py`。
- generalist diffusion baseline 在 `multi_task/eval_high_level.py` + `multi_task/eval_low_level.py`，用于根据 MIDI/goal 生成 trajectory/action。
- 不要把 single-song PPO 的 prior action 误认为来自 generalist diffusion checkpoint。

必须做的事：
1. 检查当前实验状态：`tmux ls`、`nvidia-smi`、`tail -f /home/gaoj/share4/_piano/baseline_results/logs/*.log`。
2. 如果没有 tmux 实验在跑，就启动自动 baseline：
   `cd /home/gaoj/share4/_piano/pianomime && GPU_IDS="<空闲GPU列表>" SESSION=pianomime_baseline RUN_ID=baseline_$(date +%Y%m%d) bash scripts/start_tmux_baseline.sh`
3. 如果 GPU 被别人占用，不要退出；scheduler 会等待显存低于阈值后自动领取 GPU。
4. 如果 Codex/VSCode 断开，实验也必须继续在 tmux 中运行。恢复后先 `tmux attach -t pianomime_baseline` 或查看日志。
5. 如果任务中断，使用同一个 `RUN_ID` 重新执行 `scripts/start_tmux_baseline.sh`；已有 metrics/video 会跳过，高层 trajectory 会复用，PPO 会尽量从已有 best checkpoint 继续。
6. 保留所有日志、metrics、视频、checkpoint 和必要痕迹在 `/home/gaoj/share4/_piano/baseline_results`。
7. 不要擅自改 reward、任务配置、baseline 超参或策略结构；若为了修复可运行性做工程改动，必须记录到 `docs/planning.md` 和 `docs/problems.md`。

如果无法联网，可先设置代理：
`export http_proxy=http://10.0.0.204:1080 https_proxy=http://10.0.0.204:1080`
或
`export http_proxy=http://10.0.0.204:1090 https_proxy=http://10.0.0.204:1090`

验收标准：
- single-song replay 至少有 `Stan_1`、`Petrunko_3`、`NeverGonnaGiveYouUp_1` 的 metrics 和视频。
- generalist diffusion 至少有 `Alone_1`，并继续补 `Numb_1`、`NoTimeToDie_1`。
- PPO residual 至少为 `Petrunko_3` 产出 `eval_metrics.csv`、`eval_f1_curve.png`、checkpoint/训练日志。
- 最终给出清晰状态汇报：完成了什么、每项结果在哪里、哪些问题仍未解决、下一步建议。
```
