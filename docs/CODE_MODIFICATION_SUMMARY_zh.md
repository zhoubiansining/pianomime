# 代码修改总结

最后更新：2026-05-19

本文档记录为了在课程服务器上稳定复现 PianoMime baseline 所做的工程修改。修改目标是尽量保持原始 baseline 算法不变，同时让仓库更容易运行、恢复和交给同学维护。

## 安全边界

以下 baseline 语义没有被修改：

- 没有重新设计 reward terms。
- 没有修改 model architecture。
- 没有修改 diffusion checkpoint 或 task definition。
- baseline 默认的 PPO policy architecture 和核心 hyperparameters 保持不变；这些默认值现在集中写在 `configs/baseline.toml`，便于后续新方法复制后修改。

绝大多数修改都是路径处理、依赖容错、日志记录、自动化和运行稳定性相关的工程改动。

## 修改区域总览

| 区域 | 文件 | 修改内容 |
| --- | --- | --- |
| Multi-task diffusion eval | `multi_task/eval_high_level.py`, `multi_task/eval_low_level.py` | 增加 repo-root 路径解析、提前做 CLI 参数检查、device-aware checkpoint loading、输出目录自动创建。 |
| Multi-task utilities | `multi_task/utils.py` | 延迟导入环境依赖、repo-relative dataset lookup、CPU/GPU 安全的 encoder device 使用、eval observation encoding 中使用 `torch.no_grad()`。 |
| Single-song PPO training | `single_task/train_ppo.py` | 持久化 evaluation CSV、保存 F1 curve image、支持更安全 rerun、支持 pretrained resume、final rollout guard。 |
| Single-song replay | `single_task/test_trained_actions.py`, `single_task/utils.py` | 使用 repo-relative dataset/action loading，并减少启动时的脆弱导入。 |
| Central config | `configs/baseline.toml`, `pianomime_config.py`, `scripts/config_export.py`, `scripts/run_ppo_from_config.py` | 将路径、任务列表、scheduler 默认值、single-song/generalist baseline 超参集中到 TOML 配置。 |
| Audio/video robustness | `robopianist/wrappers/sound.py` | 音频依赖变成可选；缺少 FluidSynth/PortAudio 时仍能生成 silent video。 |
| Automation scripts | `scripts/*.sh`，尤其是 `baseline_scheduler.sh`, `start_tmux_baseline.sh`, `sync_to_runtime.sh`, `run_multisong_task.sh` | tmux 运行、GPU 等待和锁定、scratch 目录执行、结果同步、任务可重复运行。 |
| Setup/release docs | `README.md`, `COURSE_BASELINE.md`, `docs/*.md`, `.gitignore`, `requirements.txt` | 增加课程文档、artifact setup、结果索引、代码审计、忽略大型下载/生成文件。 |

## 代表性修改

### 1. Central config

相关文件：

```text
configs/baseline.toml
pianomime_config.py
scripts/config_export.py
scripts/run_ppo_from_config.py
scripts/baseline_scheduler.sh
scripts/run_ppo.sh
single_task/train_ppo.py
single_task/test_trained_actions.py
multi_task/eval_high_level.py
multi_task/eval_low_level.py
```

baseline 的路径、任务列表和核心超参数现在统一从 `configs/baseline.toml` 读取：

```toml
[single_song.ppo]
total_iters = 2000
residual_factor = 0.03
ppo_batch_size = 1024
policy_activation = "gelu"
policy_pi_arch = [1024, 256]
policy_vf_arch = [1024, 256]
```

脚本入口通过同一套 loader 展开配置：

```python
cfg = load_config(args.config)
ppo = dict(section(cfg, "single_song", "ppo"))
command.extend(cli_args_from_mapping(ppo))
```

这样后续同学做新方法时可以复制一份 config，而不是在多个 shell/Python 文件里寻找分散的默认值。

### 2. Repo-root 路径处理

相关文件：

```text
multi_task/eval_high_level.py
multi_task/eval_low_level.py
single_task/train_ppo.py
single_task/test_trained_actions.py
single_task/utils.py
```

原始脚本依赖从某个特定父目录启动。现在改为从 `__file__` 解析 repository root。

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

代表性的 artifact loading 现在使用 `PROJECT_ROOT`：

```python
dataset_path = str(PROJECT_ROOT / "dataset_ll.zarr")
ckpt_path = PROJECT_ROOT / "checkpoint_low_level.ckpt"
left_hand_action_list = np.load(
    PROJECT_ROOT / "multi_task" / "trajectories" / f"{task_name}_left_hand_action_list.npy"
)
```

这样脚本可以从 tmux run directory 或 local scratch 运行，不需要用户手动切到某个固定目录。

### 3. CPU/GPU 安全的模型加载

相关文件：

```text
multi_task/eval_high_level.py
multi_task/eval_low_level.py
multi_task/utils.py
```

Checkpoint loading 现在跟随实际可用 device：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
```

Observation encoding 不再假设一定可以 `.cuda()`：

```python
def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

device = _module_device(encoder)
goal = torch.from_numpy(goal).to(device)
with torch.no_grad():
    goal = encoder.forward_without_sampling(goal)
```

这可以避免 CPU-only fallback 直接崩溃，也避免 eval 时构建不必要的 autograd graph。

### 4. Lazy environment imports

相关文件：

```text
multi_task/utils.py
```

MuJoCo/RoboPianist 等重依赖被延迟到真正构建环境时再导入：

```python
def _env_deps():
    from robopianist.suite.tasks import piano_with_shadow_hands_res
    from dm_env_wrappers import CanonicalSpecWrapper, SinglePrecisionWrapper, DmControlWrapper
    from robopianist.wrappers import PianoSoundVideoWrapper
    from robopianist.wrappers.deep_mimic import DeepMimicWrapper
    from robopianist.wrappers.fingering_emb import FingeringEmbWrapper
    from robopianist.wrappers.residual import ResidualWrapper
    ...
```

这让轻量 import 和错误参数检查更快，也减少 headless server 上导入时失败的概率。

### 5. Train/test note trajectory lookup

相关文件：

```text
multi_task/utils.py
```

Generalist evaluation 可以显式查找 training 或 test note trajectories，不再使用过宽的异常吞掉逻辑：

```python
def _load_note_trajectory(task_name):
    train_path = PROJECT_ROOT / "dataset" / "notes" / f"{task_name}.pkl"
    test_path = PROJECT_ROOT / "dataset" / "notes_test" / f"{task_name}.pkl"
    path = train_path if train_path.exists() else test_path
    with path.open("rb") as f:
        return pickle.load(f)
```

这对 unseen-song evaluation 很重要，因为 test songs 位于 `dataset/notes_test`。

### 6. PPO training metrics 和 F1 curve

相关文件：

```text
single_task/train_ppo.py
```

现在 baseline 会留下 machine-readable F1 trace：

```python
def append_eval_metrics(path: Path, iteration: int, env_steps: int, metrics: dict) -> None:
    row = {
        "iteration": iteration,
        "env_steps": env_steps,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "sustain_precision": metrics.get("sustain_precision"),
        "sustain_recall": metrics.get("sustain_recall"),
        "sustain_f1": metrics.get("sustain_f1"),
    }
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row)).writerow(row)
```

训练结束或中断时都会尝试画曲线：

```python
finally:
    maybe_plot_f1_curve(metrics_path, f1_curve_path)
```

产物包括：

```text
eval_metrics.csv
eval_f1_curve.png
```

### 7. PPO rerun 和中断恢复

相关文件：

```text
single_task/train_ppo.py
scripts/baseline_scheduler.sh
```

Experiment directory 会复用，不会因为目录已存在就失败：

```python
experiment_dir.mkdir(parents=True, exist_ok=True)
if not metrics_path.exists():
    write_eval_metrics_header(metrics_path)
```

Scheduler 可以从已有 best checkpoint 恢复：

```bash
checkpoint_zip="$run_dir/robopianist_rl/ckpts/${run_name}_best.zip"
maybe_pretrained=()
if [[ -f "$checkpoint_zip" ]]; then
  maybe_pretrained=(--pretrained "$checkpoint_zip")
fi
```

如果中断前没有产生 checkpoint，final rollout 会安全跳过：

```python
if not checkpoint_path.with_suffix(".zip").exists():
    print(f"No checkpoint produced at {checkpoint_path.with_suffix('.zip')}; skip final rollout.")
    return
```

### 8. Silent-video fallback

相关文件：

```text
robopianist/wrappers/sound.py
```

服务器缺少系统级 FluidSynth/PortAudio libraries。现在 video wrapper 在缺少音频依赖时会退化为 silent video，而不是让整个实验失败：

```python
try:
    from robopianist.music import synthesizer
    self._synth = synthesizer.Synthesizer(sf2_path, sample_rate)
except ImportError as exc:
    self._synth = None
    self._synth_error = exc

def _write_frames(self) -> None:
    super()._write_frames()
    if self._synth is None:
        return
```

这样即使没有音频支持，也能保留可用于视觉检查的 baseline videos。

### 9. tmux scheduler 和 GPU waiting

相关文件：

```text
scripts/start_tmux_baseline.sh
scripts/baseline_scheduler.sh
scripts/sync_to_runtime.sh
scripts/run_multisong_task.sh
```

Scheduler 会等待空闲 GPU，用 lock directory 领取 GPU，并把所有 logs/results 写到稳定共享路径：

```bash
claim_gpu() {
  local label="$1"
  while true; do
    for gpu in $GPU_IDS; do
      used="$(gpu_mem_used "$gpu" || echo 999999)"
      lock="$LOCK_DIR/gpu_${gpu}.lock"
      if [[ "$used" =~ ^[0-9]+$ ]] && (( used < GPU_FREE_MEM_MB )); then
        if mkdir "$lock" 2>/dev/null; then
          printf '%s\n' "$BASHPID" > "$lock/pid"
          printf '%s\n' "$gpu"
          return 0
        fi
      fi
    done
    sleep "$POLL_SECONDS"
  done
}
```

Scheduler 从 local scratch 运行实验：

```bash
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/gaoj/piano_scratch}"
RUNTIME_DIR="${RUNTIME_DIR:-$RUNTIME_ROOT/pianomime}"
```

这样可以避免仿真和视频写入全部压在共享文件系统上。

### 10. Artifact setup 和 Git hygiene

相关文件：

```text
scripts/setup_artifacts.sh
scripts/setup_python_env.sh
.gitignore
requirements.txt
docs/CONFIGURATION_zh.md
docs/USAGE_zh.md
```

大型下载和生成文件会被 Git 忽略：

```gitignore
dataset/
dataset_hl.zarr/
dataset_ll.zarr/
checkpoint_*.ckpt
baseline_results/
multi_task/trajectories/
*.mp4
```

Artifacts 可以用下面命令恢复：

```bash
bash scripts/setup_artifacts.sh
```

## 已做验证

- 修改后的 Python entrypoints 和 utilities 通过 `py_compile`。
- 自动化 shell scripts 通过 shell syntax check。
- RoboPianist environment reset/step smoke test 通过。
- 已生成 3 个 single-song replay videos 和 metrics。
- `Petrunko_3` PPO training 产生了 `eval_metrics.csv`、`eval_f1_curve.png` 和 final rollout video。
- 7 首 unseen-song generalist evaluations 产生了 metrics 和 videos。

## 剩余非算法 caveats

- 视频没有声音，除非安装系统级 FluidSynth/PortAudio libraries。
- 当前设置已在 A800 服务器上验证；4090 路径有 smoke script，但本机没有可见 4090，所以尚未物理验证。
- GitHub push 还需要一个空 remote repository URL。
