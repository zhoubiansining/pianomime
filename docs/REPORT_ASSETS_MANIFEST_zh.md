# 报告图片资产清单

最后更新：2026-05-27

本目录的报告小图都放在：

```text
docs/report_assets/
```

这些图片是从已完成 baseline 结果中复制或截帧得到的，方便报告、poster 或 slides 直接使用。原始视频和完整结果仍保存在：

```text
/home/gaoj/share4/_piano/baseline_results
```

## 核心图

| 文件 | 用途 |
| --- | --- |
| `ppo_petrunko_f1_curve.png` | 报告中的 PPO F1 training curve |
| `Petrunko_3_ppo_final_t5s.png` | PPO best-checkpoint rollout 的 5 秒截图 |

## Single-song 截图

| 文件 | 对应视频 |
| --- | --- |
| `Stan_1_single_song_baseline_t5s.png` | `single_song/videos/Stan_1_single_song_baseline.mp4` |
| `Petrunko_3_single_song_baseline_t5s.png` | `single_song/videos/Petrunko_3_single_song_baseline.mp4` |
| `NeverGonnaGiveYouUp_1_single_song_baseline_t5s.png` | `single_song/videos/NeverGonnaGiveYouUp_1_single_song_baseline.mp4` |

## Generalist 截图

| 文件 | 对应视频 |
| --- | --- |
| `Alone_1_multisong_t5s.png` | `multisong/videos/Alone_1_multisong_baseline.mp4` |
| `Numb_1_multisong_t5s.png` | `multisong/videos/Numb_1_multisong_baseline.mp4` |
| `NoTimeToDie_1_multisong_t5s.png` | `multisong/videos/NoTimeToDie_1_multisong_baseline.mp4` |
| `Forester_1_multisong_t5s.png` | `multisong/videos/Forester_1_multisong_baseline.mp4` |
| `EyesClosed_1_multisong_t5s.png` | `multisong/videos/EyesClosed_1_multisong_baseline.mp4` |
| `Paradise_1_multisong_t5s.png` | `multisong/videos/Paradise_1_multisong_baseline.mp4` |
| `SomewhereOnlyWeKnow_1_multisong_t5s.png` | `multisong/videos/SomewhereOnlyWeKnow_1_multisong_baseline.mp4` |

## 建议展示选择

- 报告正文：`ppo_petrunko_f1_curve.png`。
- Single-song 展示：`Stan_1`、`Petrunko_3`、`NeverGonnaGiveYouUp_1` 三张截图任选 1-3 张。
- Generalist 展示：建议选 `NoTimeToDie_1`、`EyesClosed_1`、`Paradise_1`、`Forester_1`、`Alone_1` 五首，和后续 improvement 对比要求保持一致。
