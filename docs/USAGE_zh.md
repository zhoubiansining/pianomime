# 使用说明

本文档说明课程维护版 baseline workflow。路径默认共享服务器目录挂载在 `/home/gaoj/share4`。

## 目录结构

```text
/home/gaoj/share4/_piano/
  pianomime/                 # 共享源码目录
  .venv/                     # 共享 Python 环境
  artifacts/                 # Google Drive zip 缓存
  baseline_results/          # 同步后的日志、metrics、视频、训练输出

/home/gaoj/piano_scratch/
  pianomime/                 # 本机运行副本，用于减少共享盘 I/O
  baseline_results/          # 本机结果暂存
  runs/                      # 本机运行目录
```

共享源码目录是 source of truth。长实验应该先通过 `scripts/sync_to_runtime.sh` 同步到本地 scratch，再从 scratch 运行。

如果只是查看已经完成的 baseline metrics、videos 和 training curve，请先读 `docs/BASELINE_RESULTS_zh.md`。除非在验证新环境，否则不需要重复跑已完成 baseline。

## 配置文件

默认配置文件是：

```text
configs/baseline.toml
```

路径、artifacts、GPU scheduler、single-song replay、PPO residual training、generalist high-level/low-level evaluation 的核心参数都在这里。需要改曲目、结果目录或超参时，建议复制一份新配置：

```bash
cp configs/baseline.toml configs/my_method.toml
CONFIG_FILE=configs/my_method.toml bash scripts/run_multisong_task.sh Alone_1 0
```

详细说明见 `docs/CONFIGURATION_zh.md`。

## 环境

如果已有环境，直接激活：

```bash
source /home/gaoj/share4/_piano/.venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

如果环境缺失，重建：

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_python_env.sh
```

如果联网失败，可以先设置代理：

```bash
export http_proxy=http://10.0.0.204:1080 https_proxy=http://10.0.0.204:1080
# 或
export http_proxy=http://10.0.0.204:1090 https_proxy=http://10.0.0.204:1090
```

## 数据和 Checkpoints

准备 dataset 和 checkpoints：

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/setup_artifacts.sh
```

脚本会优先复用 `/home/gaoj/share4/_piano/artifacts` 里的缓存 zip。只有缓存不存在时才会从原始 PianoMime Google Drive 链接下载。

仓库里已经包含 `dataset_hl.zarr` 和 `dataset_ll.zarr`，所以 generalist multi-song baseline 不再需要额外从共享目录补拷这两个数据目录。

## Smoke Check

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/check_4090_feasibility.sh
```

虽然脚本名里有 4090，但它本质上是通用的 CUDA/RoboPianist smoke test。在 4090 上运行时，它会检查 CUDA allocation 和 headless import 是否正常。

## 手动 Baseline 命令

优先使用下面这些脚本入口；它们会读取 `CONFIG_FILE` 指定的 TOML 配置。

Single-song action replay：

```bash
bash scripts/test_trained_actions.sh Stan_1
```

Generalist diffusion baseline：

```bash
bash scripts/run_multisong_task.sh Alone_1 0
```

PPO residual baseline：

```bash
bash scripts/run_ppo.sh Petrunko_3
```

无人值守运行时，优先使用 `docs/EXPERIMENT_AUTOMATION_zh.md` 中的 tmux scheduler。

当前已复现结果可以这样查看：

```bash
ls /home/gaoj/share4/_piano/baseline_results/single_song/videos
ls /home/gaoj/share4/_piano/baseline_results/multisong/videos
```
