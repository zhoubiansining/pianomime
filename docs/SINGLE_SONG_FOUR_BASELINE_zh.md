# 四首 Single-Song Baseline 对齐记录

最后更新：2026-06-04

后续 single-song 改进统一使用下面四首曲目：

```text
TwinkleTwinkleRousseau
Pirates_1
Stan_1
Petrunko_3
```

这四首已经写入 `configs/baseline.toml` 的 `single_song.baseline_songs` 和 `single_song.ppo_songs`。

## 当前状态

| Song | Action replay baseline | PPO residual baseline |
| --- | --- | --- |
| `TwinkleTwinkleRousseau` | 原始 release 缺少 low-level actions，不能 replay | 已完成，best-checkpoint rollout F1 0.7912 |
| `Pirates_1` | 原始 release 缺少 low-level actions，不能 replay | 已完成，best-checkpoint rollout F1 0.8718 |
| `Stan_1` | 已有，F1 0.9795 | 旧 run 中断；可按同一配置重跑 |
| `Petrunko_3` | 已有，F1 0.8900 | 已有，best F1 0.795686 |

已有 action replay 视频：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/videos/Stan_1_single_song_baseline.mp4
/home/gaoj/share4/_piano/baseline_results/single_song/videos/Petrunko_3_single_song_baseline.mp4
```

已有 PPO 曲线：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Petrunko_3_ppo_curve_20260513_135059/eval_f1_curve.png
```

完整 2000-iteration PPO 结果：

| Song | Iterations | Env steps | Best-checkpoint rollout P/R/F1 | Last evaluation P/R/F1 | 曲线/视频目录 |
| --- | ---: | ---: | --- | --- | --- |
| `TwinkleTwinkleRousseau` | 2000 | 1,024,000 | 0.7693 / 0.7025 / 0.7912 | 0.9338 / 0.6639 / 0.6406 | `TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548` |
| `Pirates_1` | 2000 | 1,024,000 | 0.9064 / 0.8722 / 0.8718 | 0.8369 / 0.7950 / 0.8177 | `Pirates_1_ppo_curve_fix2_20260603_114548` |

结果文件：

```text
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548/eval_metrics.csv
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548/eval_f1_curve.png
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/TwinkleTwinkleRousseau_ppo_curve_fix2_20260603_114548/eval/02001.mp4

/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548/eval_metrics.csv
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548/eval_f1_curve.png
/home/gaoj/share4/_piano/baseline_results/single_song/training_runs/Pirates_1_ppo_curve_fix2_20260603_114548/eval/02001.mp4
```

## Smoke Test 结果

这两个 smoke test 使用 `total_iters = 1`，只用于验证 pipeline 修复是否有效，不作为最终 baseline 数字。

| Song | Env steps | Precision | Recall | F1 | Video |
| --- | ---: | ---: | ---: | ---: | --- |
| `TwinkleTwinkleRousseau` | 512 | 0.1876 | 0.6959 | 0.4196 | `/home/gaoj/piano_scratch/baseline_results/single_song/smoke_runs/TwinkleTwinkleRousseau_smoke_fix_20260603/eval/00002.mp4` |
| `Pirates_1` | 512 | 0.8697 | 0.8036 | 0.8304 | `/home/gaoj/piano_scratch/baseline_results/single_song/smoke_runs/Pirates_1_smoke_fix_20260603/eval/00002.mp4` |

## 修复内容

- `TwinkleTwinkleRousseau`
  - 原因：原始 default `control_timestep = 0.05` 时，内置 MIDI task 展开为 451 steps，而现有 fingertip demo trajectory 是 150 帧。
  - 修复：为该曲目设置 `control_timestep = 0.15`，使 task 长度变为 151 steps，并将 demo 末帧 padding 一帧。
- `Pirates_1`
  - 原因：左手 residual prior 初始化时，`quadprog` 对该 IK/QP 返回空解，但同一问题可由 `daqp/osqp/scs` 求解。
  - 修复：`single_task/controller/qp_solver.py` 中增加 solver fallback，不再把 `quadprog` 的数值失败误判为轨迹不可行。
- `single_task/utils.py`
  - 增加 demonstration 与 task 长度对齐逻辑。
  - 将 `demo_ctrl_timestep` 显式设为当前 `control_timestep`，避免 per-song timestep 改动后 demo index 跳帧。
