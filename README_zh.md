# PianoMime：从互联网示范中学习通用、灵巧的钢琴演奏机器人
[[项目主页]](https://pianomime.github.io/)
[[论文 PDF]](https://arxiv.org/pdf/2407.18178)
[[Arxiv]](https://arxiv.org/abs/2407.18178)
[[Colab]](https://colab.research.google.com/drive/1Rv1XGPA0a4x3a_M6yXc7uiwKnmmIu95o?usp=sharing)

## 课程 Baseline 分支

这个 fork 保留了原始 PianoMime 代码库，并额外补充了适合课程作业交接和复现的 baseline workflow，包括复现实验脚本、无图形界面服务器适配、集中式配置和适合长时间运行的 tmux 自动化。

### 课程中维护的内容

- 单曲回放 baseline：`scripts/test_trained_actions.sh`
- 单曲 PPO baseline：`scripts/run_ppo.sh`
- 通用 checkpoint 评测：`scripts/run_multisong_task.sh`
- 多任务自动调度入口：`scripts/start_tmux_baseline.sh`
- 路径、曲目和超参数的集中配置：`configs/baseline.toml`

当前已经复现出的 baseline 结果包括 3 个训练集单曲视频、1 条 `Petrunko_3` 的 PPO F1 训练曲线，以及 7 个未见曲目的 generalist evaluation 视频。具体指标和输出路径整理在：

- `docs/BASELINE_RESULTS.md`
- `docs/BASELINE_RESULTS_zh.md`
- `docs/BASELINE_REPORT_MATERIALS_zh.md`
- `docs/BASELINE_REPORT_SECTION.tex`
- `docs/REPORT_ASSETS_MANIFEST_zh.md`
- `docs/NEXT_STEPS_zh.md`

### 快速开始

下面这套流程是推荐给课程组员的接手路径。

1. 克隆仓库并进入项目目录：

   ```bash
   git clone https://github.com/zhoubiansining/pianomime.git
   cd pianomime
   ```

2. 安装系统依赖：

   ```bash
   bash scripts/install_deps.sh
   ```

   在 Linux 上，这一步会安装 `ffmpeg`、`fluidsynth`、`portaudio` 等相关依赖。仓库里已经包含打包好的 Shadow Hand 资产和默认 soundfont，所以 fresh clone 后不需要再单独准备 `third_party` 目录。

3. 创建 Python 环境：

   ```bash
   bash scripts/setup_python_env.sh
   ```

   这个脚本会根据 `configs/baseline.toml` 创建 virtualenv，安装 CUDA 版 PyTorch，然后安装 `requirements.txt` 中的依赖。

4. 下载官方数据集和发布的 checkpoints：

   ```bash
   bash scripts/setup_artifacts.sh
   ```

   仓库已经包含 `dataset_hl.zarr` 和 `dataset_ll.zarr`，因此 multi-song baseline 不再需要额外从共享目录手动补拷数据。

5. 运行 smoke test：

   ```bash
   bash scripts/check_4090_feasibility.sh
   ```

6. 运行一条 baseline 命令：

   ```bash
   bash scripts/test_trained_actions.sh Stan_1
   # 或
   bash scripts/run_multisong_task.sh Alone_1 0
   # 或
   bash scripts/run_ppo.sh Petrunko_3
   ```

### 组员阅读顺序

- 中文交接总览：`COURSE_BASELINE_zh.md`
- 英文交接总览：`COURSE_BASELINE.md`
- 使用说明：`docs/USAGE.md`、`docs/USAGE_zh.md`
- 配置说明：`docs/CONFIGURATION.md`、`docs/CONFIGURATION_zh.md`
- tmux 自动化：`docs/EXPERIMENT_AUTOMATION.md`、`docs/EXPERIMENT_AUTOMATION_zh.md`
- 当前已知注意事项：`docs/problems.md`、`docs/problems_zh.md`

### 范围说明

- 课程维护的主要工作流集中在上面的几个脚本入口，以及原项目发布的 checkpoints。
- `multi_task/`、`goal_auto_encoder/` 和 `tutorial/` 下的一些研究或训练代码仍然保留，供参考使用，但不是课程交接的主入口。
- 数据预处理 notebook 可能还需要 baseline 环境之外的额外依赖，例如 MediaPipe。

## 原始项目背景

**Cheng Qian**<sup>1</sup>, **Julen Urain**<sup>2</sup>, **Kevin Zakka**<sup>3</sup>, **Jan Peters**<sup>2</sup>

<sup>1</sup>TU Munich,
<sup>2</sup>TU Darmstadt,
<sup>3</sup>UC Berkeley

简述：
我们训练了一个通用策略，用灵巧机械手根据互联网上的人类钢琴演奏示范来演奏任意曲目。方法上，我们用 residual reinforcement learning 学习针对特定曲目的策略，再用两阶段 diffusion policy 泛化到新曲目。

[![Video](https://i.ytimg.com/vi/LW0AiBIcnL0/hqdefault.jpg)](https://youtu.be/LW0AiBIcnL0)

## 数据预处理教程

我们提供了一个 notebook，用于从视频和 MIDI 文件中准备训练数据：

[数据预处理教程](tutorial/data_preprocessing.ipynb)

在这个 notebook 里，你可以学习如何：

- 估计从视频坐标到钢琴坐标的单应矩阵
- 从视频中提取指法标签和指尖轨迹
- 将处理后的数据整理成可训练格式

我们也提供了 Google Colab 版本：
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rv1XGPA0a4x3a_M6yXc7uiwKnmmIu95o?usp=sharing)

## 引用

如果你在学术场景中使用本项目，请引用：

```bibtex
@misc{qian2024pianomimelearninggeneralistdexterous,
      title={PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations},
      author={Cheng Qian and Julen Urain and Kevin Zakka and Jan Peters},
      year={2024},
      eprint={2407.18178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.18178},
}
```

## 致谢

仿真环境基于 [RoboPianist](https://github.com/google-research/robopianist)。

Diffusion policy 部分改编自 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)。

逆运动学控制器改编自 [Pink](https://github.com/stephane-caron/pink)。

人类演奏示范视频来自 YouTube 频道 [PianoX](https://www.youtube.com/channel/UCsR6ZEA0AbBhrF-NCeET6vQ)。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
