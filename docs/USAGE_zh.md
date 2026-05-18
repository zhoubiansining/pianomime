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

## Smoke Check

```bash
cd /home/gaoj/share4/_piano/pianomime
bash scripts/check_4090_feasibility.sh
```

虽然脚本名里有 4090，但它本质上是通用的 CUDA/RoboPianist smoke test。在 4090 上运行时，它会检查 CUDA allocation 和 headless import 是否正常。

## 手动 Baseline 命令

Single-song action replay：

```bash
cd /home/gaoj/piano_scratch/runs/single_Stan_1
export PYTHONPATH=/home/gaoj/piano_scratch/pianomime:/home/gaoj/piano_scratch/pianomime/single_task
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl
/home/gaoj/share4/_piano/.venv/bin/python \
  /home/gaoj/piano_scratch/pianomime/single_task/test_trained_actions.py Stan_1
```

Generalist diffusion baseline：

```bash
cd /home/gaoj/piano_scratch/runs/multisong_Alone_1
export PYTHONPATH=/home/gaoj/piano_scratch/pianomime
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl
/home/gaoj/share4/_piano/.venv/bin/python \
  /home/gaoj/piano_scratch/pianomime/multi_task/eval_high_level.py Alone_1
/home/gaoj/share4/_piano/.venv/bin/python \
  /home/gaoj/piano_scratch/pianomime/multi_task/eval_low_level.py Alone_1
```

PPO residual baseline 使用原始任务超参：

```bash
bash scripts/run_ppo.sh Petrunko_3
```

无人值守运行时，优先使用 `docs/EXPERIMENT_AUTOMATION_zh.md` 中的 tmux scheduler。

当前已复现结果可以这样查看：

```bash
ls /home/gaoj/share4/_piano/baseline_results/single_song/videos
ls /home/gaoj/share4/_piano/baseline_results/multisong/videos
```
