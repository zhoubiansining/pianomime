# 配置说明

最后更新：2026-05-19

baseline pipeline 的主要路径、任务列表和超参数已经集中到：

```text
configs/baseline.toml
```

默认脚本都会读取这份配置。需要切换配置时，使用 `CONFIG_FILE`：

```bash
CONFIG_FILE=configs/my_method.toml bash scripts/start_tmux_baseline.sh
CONFIG_FILE=configs/my_method.toml bash scripts/run_multisong_task.sh Alone_1 0
CONFIG_FILE=configs/my_method.toml bash scripts/run_ppo.sh Petrunko_3
```

## 配置结构

| 配置段 | 作用 |
| --- | --- |
| `[paths]` | 项目、venv、artifacts、结果目录、本地 scratch 目录 |
| `[artifacts]` | 官方 dataset/checkpoint 下载 id 和缓存文件名 |
| `[environment]` | MuJoCo、JAX/W&B 等运行环境变量 |
| `[scheduler]` | tmux session、GPU 显存阈值、轮询间隔 |
| `[single_song]` | single-song replay 和 PPO 默认曲目列表 |
| `[single_song.replay]` | replay baseline 的环境、视频、reward wrapper 参数 |
| `[single_song.ppo]` | PPO residual baseline 的训练超参和网络结构 |
| `[multisong]` | generalist baseline 默认测试曲目 |
| `[multisong.high_level]` | high-level diffusion evaluation 参数 |
| `[multisong.low_level]` | low-level diffusion evaluation 参数 |

## 路径占位符

`configs/baseline.toml` 支持以下占位符：

```text
{project_root}     当前仓库根目录
{project_parent}   当前仓库父目录
{shared_root}      [paths.shared_root]
{runtime_root}     [paths.runtime_root]
{venv}             [paths.venv]
{results_dir}      [paths.results_dir]
{local_results_dir} [paths.local_results_dir]
```

默认情况下，如果仓库位于 `/home/gaoj/share4/_piano/pianomime`，配置会展开为：

```text
shared_root       = /home/gaoj/share4/_piano
venv              = /home/gaoj/share4/_piano/.venv
results_dir       = /home/gaoj/share4/_piano/baseline_results
runtime_root      = /home/gaoj/piano_scratch
runtime_dir       = /home/gaoj/piano_scratch/pianomime
local_results_dir = /home/gaoj/piano_scratch/baseline_results
```

## 推荐改法

复现 baseline 时不要改 `configs/baseline.toml`。

做新方法时建议复制一份配置：

```bash
cp configs/baseline.toml configs/my_method.toml
```

然后只在新配置中修改曲目、checkpoint、训练超参或结果目录。例如：

```toml
[single_song.ppo]
total_iters = 3000
initial_lr = 0.0001
seed = 123
```

运行时指定：

```bash
CONFIG_FILE=configs/my_method.toml RUN_ID=my_method_seed123 bash scripts/run_ppo.sh Petrunko_3
```

这样 baseline 配置、baseline 结果和新方法结果可以清楚分开，方便后续汇报和复查。
