# PianoMime 问题记录

最后更新：2026-06-03

本文档区分已经解决或缓解的问题，以及后续同学仍然需要注意的 caveats。

## 已解决或已缓解

1. 缺少 dataset 和 checkpoints。
   - 已通过恢复官方 PianoMime artifacts 解决。
   - 后续设置方式：`bash scripts/setup_artifacts.sh`。

2. Python 依赖缺失或版本不兼容。
   - 当前服务器已通过 `/home/gaoj/share4/_piano/.venv` 解决。
   - `requirements.txt` 记录了当前测试过的 package pins。
   - CUDA PyTorch 由 `scripts/setup_python_env.sh` 安装。

3. 原始环境中的 PyTorch 不能使用 CUDA。
   - 已在项目 virtualenv 中用 CUDA 11.8 PyTorch wheels 解决。

4. 部分脚本依赖脆弱的当前工作目录。
   - 主要 eval/replay/training entrypoints 现在通过 `PROJECT_ROOT` 解析路径。

5. Multi-task eval 假设一定能 `.cuda()`。
   - 已通过 device-aware checkpoint loading 和 encoder execution 修复。

6. Evaluation observation encoding 中不必要地构建 autograd graph。
   - 已在代表性 encoder 调用中使用 `torch.no_grad()`。

7. Audio dependency import failures。
   - 已通过 silent video fallback 缓解；缺少 FluidSynth/PortAudio 系统库时不再导致视频生成失败。

8. 缺少 PPO training curve logs。
   - 已在 `single_task/train_ppo.py` 中加入 `eval_metrics.csv` 和 `eval_f1_curve.png` 输出。

9. 共享文件系统在长实验中可能卡住。
   - 已通过从 `/home/gaoj/piano_scratch/pianomime` 运行，并把最终结果同步回 `/home/gaoj/share4/_piano/baseline_results` 缓解。

10. 断开 VSCode 或终端后实验可能停止。
    - 已通过 tmux automation 和可重复运行的 scheduler 逻辑缓解。

## 仍需注意

1. 视频没有声音。
   - 原因：缺少系统级 FluidSynth/PortAudio libraries。
   - 影响：视频可用于视觉检查，但没有 audio track。
   - 如果展示必须有声音，需要安装系统音频包并重渲染选定视频。

2. 4090 可行性尚未在当前机器上物理验证。
   - 从模型和 checkpoint 尺寸看，一张 24 GB 4090 跑一个 job 应该可行。
   - 在真实 4090 机器上声称已验证前，应先运行 `scripts/check_4090_feasibility.sh`。

3. GitHub fresh clone 尚未完整测试。
   - 这一步依赖新的空 GitHub remote URL。
   - push 后建议在新目录 clone，然后运行 `scripts/setup_artifacts.sh`、`scripts/check_4090_feasibility.sh` 和一个短 eval command。

4. 部分系统级依赖不在 `requirements.txt` 中。
   - MuJoCo EGL、ffmpeg 和可选音频库依赖服务器系统环境。
   - `docs/USAGE_zh.md` 记录了需要的环境变量和设置方式。

5. 算法改进尚未开始。
   - Baseline reproduction 已完成。
   - 下一阶段是提升 single-song 和 generalist F1，而不是继续清理 baseline。

6. 四首 single-song 对齐集合中有两首原本需要额外工程修复后才能跑通。
   - 统一集合：`TwinkleTwinkleRousseau`、`Pirates_1`、`Stan_1`、`Petrunko_3`。
   - `TwinkleTwinkleRousseau`：现有 fingertip demo trajectory 是 150 帧；已通过 per-song `control_timestep = 0.15` 和末帧 padding 对齐到 151 个 task steps。
   - `Pirates_1`：`quadprog` 在左手 IK/QP 上数值失败；已为 QP solver 增加 `daqp/osqp/scs/ecos` fallback。
   - 两项修复已经进入仓库，完整 2000-iteration baseline 已经跑完。
   - 最终 PPO best-checkpoint rollout F1：`TwinkleTwinkleRousseau` 为 0.7912，`Pirates_1` 为 0.8718。

## 不要忘记

- 不要把 single-song PPO 的 prior action 描述成来自 diffusion generalist。它来自 stored demonstration trajectories 加 IK/QP。
- 跑 baseline comparison 时，不要随意改 reward terms、task definitions 或 baseline hyperparameters。
- 新实验输出要保留 CSV metrics、logs 和 videos，保证可追溯。
