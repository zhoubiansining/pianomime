# Reward Shaping Sweep Summary

## Configuration

- **Songs:** `TwinkleTwinkleRousseau`
- **Variants:** `baseline, fdist_0.1, fdist_0.5, fdist_1.0`
- **Seeds:** `42`
- **Total iters:** `2000`
- **Num envs:** `8`
- **Residual factor:** `0.03`

## Aggregate Results (sorted by avg F1)

| variant | runs | avg_f1 | avg_precision | avg_recall | avg_reward | avg_ep_len |
|---|---|---:|---:|---:|---:|---:|
| fdist_1.0 | 1 | 0.7036 | 0.7805 | 0.7265 | 918.79 | n/a |
| fdist_0.5 | 1 | 0.6936 | 0.8803 | 0.7095 | 912.55 | n/a |
| baseline | 1 | 0.6903 | 0.9501 | 0.7143 | 899.95 | n/a |
| fdist_0.1 | 1 | 0.6571 | 0.9690 | 0.6863 | 902.64 | n/a |

## Per-Run Results

| variant | song | seed | status | f1 | precision | recall | reward | best_video |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | TwinkleTwinkleRousseau | 42 | ok | 0.6903 | 0.9501 | 0.7143 | 899.95 | [video](eval_best.mp4) |
| fdist_0.1 | TwinkleTwinkleRousseau | 42 | ok | 0.6571 | 0.9690 | 0.6863 | 902.64 | [video](eval_best.mp4) |
| fdist_0.5 | TwinkleTwinkleRousseau | 42 | ok | 0.6936 | 0.8803 | 0.7095 | 912.55 | [video](eval_best.mp4) |
| fdist_1.0 | TwinkleTwinkleRousseau | 42 | ok | 0.7036 | 0.7805 | 0.7265 | 918.79 | [video](eval_best.mp4) |
