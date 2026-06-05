# Reward Shaping Sweep Summary

## Configuration

- **Songs:** `TwinkleTwinkleRousseau, Pirates_1, Stan_1, Petrunko_3`
- **Variants:** `baseline, best_timing, best_preposition, best_vel_smooth, best_finger_dist, best_top3_combo, best_full_combo`
- **Seeds:** `42`
- **Total iters:** `2000`
- **Num envs:** `8`
- **Residual factor:** `0.03`

## Aggregate Results (sorted by avg F1)

| variant | runs | avg_f1 | avg_precision | avg_recall | avg_reward | avg_ep_len |
|---|---|---:|---:|---:|---:|---:|
| best_full_combo | 4 | 0.6823 | 0.8997 | 0.6712 | 1606.98 | n/a |
| best_timing | 4 | 0.6810 | 0.9440 | 0.6689 | 1519.38 | n/a |
| best_top3_combo | 4 | 0.6639 | 0.9362 | 0.6527 | 1949.90 | n/a |
| best_vel_smooth | 4 | 0.6587 | 0.8992 | 0.6470 | 938.58 | n/a |
| baseline | 4 | 0.6499 | 0.9439 | 0.6400 | 1273.88 | n/a |
| best_preposition | 4 | 0.6431 | 0.9205 | 0.6317 | 1383.27 | n/a |
| best_finger_dist | 4 | 0.6424 | 0.8923 | 0.6348 | 1571.19 | n/a |

## Per-Run Results

| variant | song | seed | status | f1 | precision | recall | reward | best_video |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | TwinkleTwinkleRousseau | 42 | ok | 0.6903 | 0.9501 | 0.7143 | 899.95 | [video](eval_best.mp4) |
| baseline | Pirates_1 | 42 | ok | 0.7204 | 0.9919 | 0.6928 | 1593.78 | [video](eval_best.mp4) |
| baseline | Stan_1 | 42 | ok | 0.6202 | 0.9424 | 0.5838 | 1306.91 | [video](eval_best.mp4) |
| baseline | Petrunko_3 | 42 | ok | 0.5686 | 0.8910 | 0.5689 | 1294.88 | [video](eval_best.mp4) |
| best_timing | TwinkleTwinkleRousseau | 42 | ok | 0.7130 | 0.9800 | 0.7186 | 941.89 | [video](eval_best.mp4) |
| best_timing | Pirates_1 | 42 | ok | 0.7418 | 0.9953 | 0.7156 | 2030.04 | [video](eval_best.mp4) |
| best_timing | Stan_1 | 42 | ok | 0.6658 | 0.9205 | 0.6351 | 1600.59 | [video](eval_best.mp4) |
| best_timing | Petrunko_3 | 42 | ok | 0.6032 | 0.8803 | 0.6062 | 1505.01 | [video](eval_best.mp4) |
| best_preposition | TwinkleTwinkleRousseau | 42 | ok | 0.7055 | 0.8514 | 0.7302 | 901.18 | [video](eval_best.mp4) |
| best_preposition | Pirates_1 | 42 | ok | 0.6969 | 0.9914 | 0.6683 | 1752.21 | [video](eval_best.mp4) |
| best_preposition | Stan_1 | 42 | ok | 0.6078 | 0.9559 | 0.5636 | 1438.19 | [video](eval_best.mp4) |
| best_preposition | Petrunko_3 | 42 | ok | 0.5624 | 0.8833 | 0.5647 | 1441.48 | [video](eval_best.mp4) |
| best_vel_smooth | TwinkleTwinkleRousseau | 42 | ok | 0.6988 | 0.8004 | 0.7199 | 648.26 | [video](eval_best.mp4) |
| best_vel_smooth | Pirates_1 | 42 | ok | 0.7072 | 0.9869 | 0.6786 | 1152.38 | [video](eval_best.mp4) |
| best_vel_smooth | Stan_1 | 42 | ok | 0.6333 | 0.9398 | 0.5859 | 1048.77 | [video](eval_best.mp4) |
| best_vel_smooth | Petrunko_3 | 42 | ok | 0.5956 | 0.8696 | 0.6034 | 904.89 | [video](eval_best.mp4) |
| best_finger_dist | TwinkleTwinkleRousseau | 42 | ok | 0.7036 | 0.7805 | 0.7265 | 918.79 | [video](eval_best.mp4) |
| best_finger_dist | Pirates_1 | 42 | ok | 0.6957 | 0.9808 | 0.6767 | 2065.90 | [video](eval_best.mp4) |
| best_finger_dist | Stan_1 | 42 | ok | 0.6013 | 0.9286 | 0.5620 | 1675.00 | [video](eval_best.mp4) |
| best_finger_dist | Petrunko_3 | 42 | ok | 0.5691 | 0.8791 | 0.5739 | 1625.08 | [video](eval_best.mp4) |
| best_top3_combo | TwinkleTwinkleRousseau | 42 | ok | 0.7029 | 0.9701 | 0.7051 | 1044.09 | [video](eval_best.mp4) |
| best_top3_combo | Pirates_1 | 42 | ok | 0.7158 | 0.9964 | 0.6883 | 2671.14 | [video](eval_best.mp4) |
| best_top3_combo | Stan_1 | 42 | ok | 0.6334 | 0.9106 | 0.6082 | 2098.84 | [video](eval_best.mp4) |
| best_top3_combo | Petrunko_3 | 42 | ok | 0.6033 | 0.8677 | 0.6094 | 1985.52 | [video](eval_best.mp4) |
| best_full_combo | TwinkleTwinkleRousseau | 42 | ok | 0.7214 | 0.8137 | 0.7373 | 821.13 | [video](eval_best.mp4) |
| best_full_combo | Pirates_1 | 42 | ok | 0.7353 | 0.9847 | 0.7081 | 2186.03 | [video](eval_best.mp4) |
| best_full_combo | Stan_1 | 42 | ok | 0.6608 | 0.9278 | 0.6236 | 1825.32 | [video](eval_best.mp4) |
| best_full_combo | Petrunko_3 | 42 | ok | 0.6115 | 0.8726 | 0.6159 | 1595.44 | [video](eval_best.mp4) |
