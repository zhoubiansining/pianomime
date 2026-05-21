# Configuration

Last updated: 2026-05-19

The course baseline pipeline now reads its main paths, task lists, and baseline
hyperparameters from:

```text
configs/baseline.toml
```

Use `CONFIG_FILE` to run with another config:

```bash
CONFIG_FILE=configs/my_method.toml bash scripts/start_tmux_baseline.sh
CONFIG_FILE=configs/my_method.toml bash scripts/run_multisong_task.sh Alone_1 0
CONFIG_FILE=configs/my_method.toml bash scripts/run_ppo.sh Petrunko_3
```

## Sections

| Section | Purpose |
| --- | --- |
| `[paths]` | Repository, venv, artifacts, result directories, local scratch directories |
| `[artifacts]` | Official dataset/checkpoint download ids and cache names |
| `[environment]` | MuJoCo, JAX, and W&B runtime environment defaults |
| `[scheduler]` | tmux session, GPU memory threshold, polling interval |
| `[single_song]` | Default single-song replay and PPO task lists |
| `[single_song.replay]` | Replay baseline environment, video, and reward-wrapper parameters |
| `[single_song.ppo]` | PPO residual baseline training hyperparameters and policy network shape |
| `[multisong]` | Default generalist baseline songs |
| `[multisong.high_level]` | High-level diffusion evaluation parameters |
| `[multisong.low_level]` | Low-level diffusion evaluation parameters |

## Path Placeholders

`configs/baseline.toml` supports:

```text
{home}
{project_root}
{project_parent}
{shared_root}
{runtime_root}
{venv}
{results_dir}
{local_results_dir}
```

The default baseline config uses `{home}/piano_scratch` for local runtime
execution, so a fresh clone can reuse the same scripts under a different
username without editing the file immediately.

For new experiments, copy the baseline config instead of editing it directly:

```bash
cp configs/baseline.toml configs/my_method.toml
CONFIG_FILE=configs/my_method.toml RUN_ID=my_method_seed123 bash scripts/run_ppo.sh Petrunko_3
```

This keeps the reproduced baseline separate from future method changes.
