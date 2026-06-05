# Sensitivity Analysis Summary

Baseline F1: **0.6903**

| Group | Coef | Best F1 | Δ vs Baseline | Improved? |
|---|---|---:|---:|---|
| act_smooth | 0.001 | 0.6718 | -0.0185 | ✗ |
| act_smooth | 0.01 | 0.6693 | -0.0211 | ✗ |
| act_smooth | 0.1 | 0.6611 | -0.0292 | ✗ |
| finger_dist | 0.1 | 0.6571 | -0.0333 | ✗ |
| finger_dist | 0.5 | 0.6936 | +0.0033 | ✓ |
| finger_dist | 1.0 | 0.7036 | +0.0133 | ✓ |
| preposition | 0.1 | 0.6752 | -0.0152 | ✗ |
| preposition | 0.3 | 0.6755 | -0.0148 | ✗ |
| preposition | 0.5 | 0.6593 | -0.0310 | ✗ |
| preposition | 1.0 | 0.7055 | +0.0152 | ✓ |
| timing | 0.1 | 0.6774 | -0.0129 | ✗ |
| timing | 0.5 | 0.6948 | +0.0045 | ✓ |
| timing | 1.0 | 0.7130 | +0.0227 | ✓ |
| timing | 2.0 | 0.6965 | +0.0061 | ✓ |
| vel_smooth | 0.001 | 0.6988 | +0.0085 | ✓ |
| vel_smooth | 0.01 | 0.5676 | -0.1227 | ✗ |
| vel_smooth | 0.1 | 0.5100 | -0.1803 | ✗ |

## Best Coefficient per Group

| Group | Best Coef | Best F1 | Δ vs Baseline |
|---|---|---:|---:|
| act_smooth | 0.001 | 0.6718 | -0.0185 |
| finger_dist | 1.0 | 0.7036 | +0.0133 |
| preposition | 1.0 | 0.7055 | +0.0152 |
| timing | 1.0 | 0.7130 | +0.0227 |
| vel_smooth | 0.001 | 0.6988 | +0.0085 |
