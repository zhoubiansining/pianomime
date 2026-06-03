# 四首 Single-Song Baseline 对齐记录

最后更新：2026-06-03

后续 single-song 改进统一使用下面四首曲目：

```text
TwinkleTwinkleRousseau
Pirates_1
Stan_1
Petrunko_3
```

这四首已经写入 `configs/baseline.toml` 的 `single_song.baseline_songs` 和 `single_song.ppo_songs`。需要注意的是，原始 PianoMime release 中的 single-song action replay artifacts 并不覆盖这四首的全部曲目，因此当前 baseline 状态分成两类。

为了避免默认 scheduler 反复启动已知失败任务，`TwinkleTwinkleRousseau` 和 `Pirates_1` 暂时也写入了 `single_song.ppo_blocked_songs`。如果后续拿到匹配 artifacts 或修复 IK/QP，可以先清空环境变量 `PPO_BLOCKED_TASKS=""` 或修改配置再重跑。

## 当前可用结果

| Song | Action replay video | Action replay F1 | PPO residual curve |
| --- | --- | ---: | --- |
| `Stan_1` | 已有 | 0.9795 | 正在补跑：`Stan_1_ppo_curve_20260603_110918` |
| `Petrunko_3` | 已有 | 0.8900 | 已有：best F1 0.795686 |
| `TwinkleTwinkleRousseau` | 暂无 released low-level actions | 暂无 | smoke test 未通过，见下文 |
| `Pirates_1` | 暂无 released low-level actions | 暂无 | smoke test 未通过，见下文 |

已有 action replay 视频：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/videos/Stan_1_single_song_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/single_song/videos/Petrunko_3_single_song_baseline.mp4
```

已有 PPO 曲线：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/eval_f1_curve.png
```

`Stan_1` PPO 已经在 tmux 中启动：

```bash
tmux attach -t pianomime_single_song_four
tail -f /home/gaoj/piano_scratch/baseline_results/single_song/training_runs/Stan_1_ppo_curve_20260603_110918.log
```

该任务结束后会自动把 `/home/gaoj/piano_scratch/baseline_results/` 同步回 `/home/gaoj/share4/_piano/baseline_results/`。

## 未完成项及原因

`TwinkleTwinkleRousseau` 和 `Pirates_1` 目前不能直接按原始 residual single-song baseline 跑出同口径 PPO 曲线。

| Song | Smoke test log | 失败原因 |
| --- | --- | --- |
| `TwinkleTwinkleRousseau` | `/home/gaoj/piano_scratch/baseline_results/single_song/smoke_runs/TwinkleTwinkleRousseau_smoke_20260603.log` | 内置 MIDI 展开后 task note steps 为 451，但现有 fingertip demo trajectory 只有 150 帧，`DeepMimicWrapper` 要求二者长度一致，因此触发 assertion。 |
| `Pirates_1` | `/home/gaoj/piano_scratch/baseline_results/single_song/smoke_runs/Pirates_1_smoke_20260603.log` | 有 notes 和 high-level trajectory，但 residual prior 的 IK/QP 初始化时 `qp_solver.solve()` 返回空解，触发 `assert dq is not None`。 |

这些失败不是 GPU 或环境安装问题，而是曲目数据与原始 residual prior/action-replay pipeline 的匹配问题。后续如果同学已经有这两首的 processed single-song artifacts 或修复后的 IK 设置，应把对应的 notes、fingertip trajectories、trained actions/checkpoints 放入结果目录，再重新补跑。

## 代码配置变化

- `configs/baseline.toml`
  - 新增 `single_song.baseline_songs`，固定四首 single-song 对齐集合。
  - `single_song.ppo_songs` 更新为四首。
  - 为 `TwinkleTwinkleRousseau` 加了 per-song override：`use_note_trajectory = false`，因为数据包里没有 `dataset/notes/TwinkleTwinkleRousseau.pkl`。
- `scripts/run_ppo_from_config.py`
  - 支持 `[single_song.ppo.song_overrides.<song>]`，避免为了某一首歌修改全局 PPO 超参。
- `scripts/baseline_scheduler.sh`
  - single-song action replay 增加 artifacts preflight；缺少 replay actions 时记录原因并跳过，不会让整个 scheduler 崩掉。
- `pianomime_config.py`
  - `shell_default` 改为只在环境变量未设置时填默认值；现在可以用 `MULTISONG_TASKS=''` 这样的空字符串显式禁用某类任务。
