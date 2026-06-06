# Single-task PPO Optimization Record

记录日期：2026-06-01

更新日期：2026-06-05

## 数据口径

数据来自本地 `wandb` 目录：`/workspace/lwk/code/pianomime/wandb`。

本记录主要使用 `Petrunko_3` 的训练输出中打印的 `eval f1` 曲线；2026-06-05 起补充 `Pirates_2`、`Stan_1` 的跨曲目验证结果。对于增加训练步数和面向 recall 的实验，同时记录 `precision`、`recall` 和 F2，解析 `files/output.log` 里的如下日志：

```text
[ppo] iter ... | ... | f1=... | best_f1=...
```

比较口径：

- `best f1`：训练曲线中出现过的最高 `eval f1`。
- `final f1`：最后一次 eval 的 `f1`。
- `last10 mean`：最后 10 次 eval 的 `f1` 均值，用于观察末端平台表现。
- `best recall`：训练曲线中出现过的最高 `eval recall`，用于分析当前 precision 接近 1 时的 f1 瓶颈。
- `best F2`：根据 `precision` 和 `recall` 计算的最高 F2，`beta=2`，用于观察偏 recall 的 checkpoint 选择。
- `delta best`：相对 baseline best f1 的差值。

当前 `Petrunko_3` baseline：

- run：`run-20260527_213000-y7twlk0q`
- seed：42
- eval 点数：200
- best f1：0.9152
- final f1：0.9144
- last10 mean：0.91213

## 曲线汇总

| 方法 | run | eval 点数 | best f1 | delta best | final f1 | last10 mean | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | `run-20260527_213000-y7twlk0q` | 200 | 0.9152 | +0.0000 | 0.9144 | 0.91213 | 参考基线 |
| baseline_more_steps | `run-20260601_201837-si7dtltz` | 300 | 0.9175 | +0.0023 | 0.9143 | 0.91398 | 延长训练抬高峰值，但末端没有继续上升 |
| large_policy | `run-20260601_000005-yiw9ygjq` | 200 | 0.9172 | +0.0020 | 0.9172 | 0.91460 | 2000 iter 下略高于 baseline |
| large_policy_more_steps | `run-20260601_201838-yu58m9av` | 300 | 0.9211 | +0.0059 | 0.9183 | 0.91885 | 增加步数后明显超过 baseline_more_steps，是上一轮最高 |
| recall_reward_p095_n005 | `run-20260603_110330-lds3pxhv` | 200 | 0.9331 | +0.0179 | 0.9320 | 0.92964 | 当前最高 f1，0.95/0.05 权重最佳 |
| recall_reward_p090_n010 | `run-20260603_110330-2akh2erl` | 200 | 0.9327 | +0.0175 | 0.9312 | 0.92936 | 与 0.95/0.05 接近，也高于 0.85/0.15 |
| recall_reward_p085_n015 | `run-20260603_010721-n7ttogah` | 200 | 0.9322 | +0.0170 | 0.9289 | 0.92974 | 上一轮最高，仍是稳定强配置 |
| large_policy_recall_reward | `run-20260603_010720-zobk1vmo` | 200 | 0.9317 | +0.0165 | 0.9283 | 0.92833 | large policy 与 recall_reward 可叠加，接近最高组 |
| large_policy_recall_reward_p095_n005 | `run-20260604_113859-ulafj8kx` | 200 | 0.9311 | +0.0159 | 0.9280 | 0.92708 | large policy + 0.95/0.05 未超过小网络 0.95/0.05 |
| recall_reward_p099_n001 | `run-20260603_110330-bo4j6qbr` | 200 | 0.9269 | +0.0117 | 0.9232 | 0.92387 | recall 最高但 precision 明显回落，f1 不如 0.90/0.95 |
| recall_reward | `run-20260602_105823-gy66zwzt` | 200 | 0.9254 | +0.0102 | 0.9247 | 0.92389 | 第二批最高，后续被 0.85/0.15 超过 |
| recall_reward_p075_n025 | `run-20260603_010720-bab4h8ta` | 200 | 0.9250 | +0.0098 | 0.9248 | 0.92270 | 接近 0.8/0.2，但没有更高 |
| recall_missed_penalty_p060_n040_m015 | `run-20260603_010721-q0vv8v3w` | 200 | 0.9246 | +0.0094 | 0.9200 | 0.92104 | 微调 missed penalty 中最好，但 precision/recall 目标未同时满足 |
| large_policy_recall | `run-20260602_105823-2rs6i77e` | 300 | 0.9231 | +0.0079 | 0.9215 | 0.92007 | 第二批偏 recall/F2 最高 |
| recall_missed_penalty | `run-20260602_105823-s633rn6x` | 200 | 0.9220 | +0.0068 | 0.9219 | 0.92003 | 明显提高 recall，末端也稳定 |
| recall_activation_bonus | `run-20260602_105823-321rka1f` | 200 | 0.9209 | +0.0057 | 0.9198 | 0.91854 | recall 高，但 precision 损失较大 |
| recall_missed_penalty_p070_n030_m010 | `run-20260603_010720-6eqbetk7` | 200 | 0.9197 | +0.0045 | 0.9183 | 0.91541 | 降低 missed penalty 后 recall 不够，precision 仍未到 0.97 |
| recall_missed_penalty_p065_n035_m015 | `run-20260603_010720-pns0dede` | 200 | 0.9191 | +0.0039 | 0.9165 | 0.91586 | 提高 negative weight 没有恢复 precision，f1 低于原版 |
| recall_soft_wrong | `run-20260602_105823-vwicg3gh` | 200 | 0.9189 | +0.0037 | 0.9109 | 0.91543 | recall 升高，但单独使用末端回落 |
| f2_checkpoint | `run-20260602_105824-0bdqbjni` | 200 | 0.9152 | +0.0000 | 0.9144 | 0.91213 | 训练不变，只改变 checkpoint 选择，收益很小 |
| ema | `run-20260601_111455-ofit4vjo` | 200 | 0.9159 | +0.0007 | 0.9134 | 0.91301 | 略高峰值，末端接近 baseline |
| residual_reg_anneal | `run-20260601_111443-gmdrm0jj` | 200 | 0.9146 | -0.0006 | 0.9104 | 0.91086 | 基本持平，无明确提升 |
| residual_reg | `run-20260528_135801-0y5az218` | 200 | 0.9135 | -0.0017 | 0.9101 | 0.90759 | 接近 baseline，无上限提升 |
| silu_no_ortho | `run-20260601_111513-ji0qmg3k` | 200 | 0.9133 | -0.0019 | 0.9112 | 0.90940 | 接近 baseline，略低 |
| layer_norm_large | `run-20260601_111340-r8y7q1my` | 200 | 0.9113 | -0.0039 | 0.9100 | 0.90884 | 大网络加 LayerNorm 未提升 |
| actor_large | `run-20260601_111319-d4lkiydy` | 200 | 0.9106 | -0.0046 | 0.9098 | 0.90825 | 只扩 actor 不如双塔都扩 |
| advantage_clip | `run-20260531_235252-jzlna50v` | 200 | 0.9118 | -0.0034 | 0.9105 | 0.90799 | 接近 baseline，略低 |
| dual_clip | `run-20260531_235508-4dpsuu65` | 200 | 0.9118 | -0.0034 | 0.9118 | 0.90975 | 接近 baseline，略低 |
| long_horizon | `run-20260601_000038-t6s5e5to` | 200 | 0.9126 | -0.0026 | 0.9070 | 0.90922 | 完整跑完后仍无提升 |
| recall_missed_penalty_p065_n035_m010 | `run-20260603_010720-caycj3hf` | 200 | 0.9103 | -0.0049 | 0.9079 | 0.90704 | 明显负向，negative weight 提高且 penalty 降低后不稳 |
| cosine_lr | `run-20260601_111402-rl48np0i` | 200 | 0.9069 | -0.0083 | 0.9051 | 0.90265 | 低于 baseline |
| low_std | `run-20260601_111413-zcqalh0u` | 200 | 0.9073 | -0.0079 | 0.9011 | 0.90203 | 探索过低，负向 |
| clip_anneal | `run-20260531_235354-gt1lgjb1` | 200 | 0.8979 | -0.0173 | 0.8967 | 0.89433 | 明显低于 baseline |
| rsi | `run-20260528_135552-842exjeu` | 200 | 0.8829 | -0.0323 | 0.8749 | 0.87613 | 负向 |
| value_clip | `run-20260601_000121-hwfelw4m` | 200 | 0.8756 | -0.0396 | 0.8709 | 0.87012 | 负向 |
| entropy_anneal | `run-20260531_235637-zjxfrse4` | 200 | 0.8421 | -0.0731 | 0.8402 | 0.83161 | 明显负向 |
| gsde | `run-20260531_235940-4l39s6lr` | 200 | 0.6436 | -0.2716 | 0.6348 | 0.63596 | 强负向 |
| grpo | 无 Petrunko_3 run | 0 | - | - | - | - | 本地没有本曲目曲线，不能推断 |

总体判断：第一批 PPO 结构和优化器类实验中，只有 `large_policy_more_steps` 展现出明确后期优势，`best f1=0.9211`。面向 recall 的第二批实验进一步证明 reward reweight 有效，`recall_reward` 达到 `best f1=0.9254`。后续权重扫描继续提高上限：`recall_reward_p085_n015` 达到 `best f1=0.9322`，`recall_reward_p095_n005` 小幅提升到当前最高 `best f1=0.9331`。`recall_reward_p099_n001` 的 recall 最高，但 precision 掉到 `0.9684`，说明 0.99/0.01 已经过激；当前较优区间在 `0.90/0.10` 到 `0.95/0.05` 附近。large policy 与 strict recall reward 能叠加，但 `large_policy + 0.95/0.05` 没有超过小网络 `0.95/0.05`。

## 增加训练步数实验

| 方法 | total iters | best f1 / iter | precision@best f1 | recall@best f1 | best recall / iter | final f1 | final precision | final recall | last10 f1 | last10 recall | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_more_steps | 3000 | 0.9175 / 2570 | 0.9967 | 0.8844 | 0.8846 / 2560 | 0.9143 | 0.9967 | 0.8813 | 0.91398 | 0.88085 | 延长训练后峰值超过原始 baseline，但后期回落到接近原 baseline 的末端水平 |
| large_policy_more_steps | 3000 | 0.9211 / 2950 | 0.9985 | 0.8897 | 0.8897 / 2950 | 0.9183 | 0.9985 | 0.8858 | 0.91885 | 0.88659 | 后期明显优于 baseline_more_steps，且最高点接近训练末尾，说明更大网络仍在受益 |

结论：增加步数对两种方法都有帮助，但 `large_policy` 的后期收益更明显。`baseline_more_steps` 的峰值出现在 `iter=2570` 附近，之后略有回落；`large_policy_more_steps` 的峰值出现在 `iter=2950`，说明 2000 iter 时还没有充分训练。

更重要的是，这两条曲线的 precision 都接近 1：`baseline_more_steps` 峰值处 precision 为 `0.9967`，`large_policy_more_steps` 峰值处 precision 为 `0.9985`；但 recall 分别只有 `0.8844` 和 `0.8897`。因此当前 f1 的主要瓶颈不是误按太多，而是漏按目标音符。后续优化应该优先让策略愿意按下更多目标键，并接受小幅 precision 下降。

## 每种优化的思路和实际作用

### Baseline

思路：标准 PPO + residual action + DeepMimic wrapper，使用当前稳定配置作为对照。

实际作用：`best f1=0.9152`，末端 `last10 mean=0.91213`。曲线后期稳定在 0.91 左右，是当前最可靠的参考。

### RSI

思路：Reference State Initialization，在 episode reset 时从参考轨迹中间状态开始，让策略覆盖更多歌曲相位，理论上能改善长曲目的局部纠错能力。

实际作用：`best f1=0.8829`，比 baseline 低 `0.0323`。对 `Petrunko_3` 没有提高上限，反而降低最终平台。推测原因是当前 baseline 已经能从头到尾稳定跟踪，随机相位初始化破坏了从开头累积形成的状态分布，收益小于分布扰动成本。

### Residual Regularization

思路：对 residual action 加 L2 和相邻步平滑惩罚，限制过大或抖动的 residual correction，希望保留演示先验并减少无效动作。

实际作用：`best f1=0.9135`，比 baseline 低 `0.0017`，属于非常接近但没有提升。它可能让动作更保守，但 `Petrunko_3` 的上限瓶颈并不主要来自 residual 过大或不平滑，因此没有换来更高 f1。

### Residual Regularization Anneal

思路：训练早期使用较小 residual L2/smooth penalty，后期退火到 0，试图保留早期动作约束，同时避免后期精修阶段被正则化限制。

参数：`residual_l2_coef=5e-4 -> 0`，`residual_smooth_coef=5e-3 -> 0`。

实际作用：`best f1=0.9146`，比 baseline 低 `0.0006`，比固定 residual regularization 的 `0.9135` 稍好，但仍没有超过 baseline。说明“正则化后期放开”确实减轻了一部分负面影响，但本曲目上 residual 正则本身仍不是提升上限的有效方向。

### GRPO

思路：同组轨迹使用相同 reset seed，通过组内相对回报归一化做策略更新，目标是减少 value function 依赖并强化相对优劣排序。

实际作用：本地 `wandb` 中没有找到 `Petrunko_3` 的 GRPO 曲线；已有 GRPO 记录是 `Stan_1`，不能作为本曲目的效果证据。对 `Petrunko_3` 暂时只能记为“无本曲目 eval f1 证据”。

### Clip Annealing

思路：早期使用更大的 PPO clip range 增加更新幅度，后期逐步收紧 clip range，提高最终策略的保守性和精修能力。

实际作用：`best f1=0.8979`，比 baseline 低 `0.0173`。没有提升上限，后期平台明显低于 baseline。对当前任务而言，clip range 从 `0.30` 收到 `0.08` 可能过早限制了后期有效更新。

### Entropy Annealing

思路：早期增加 entropy coefficient 鼓励探索，后期降低 entropy，让策略收敛到更确定的动作。

实际作用：`best f1=0.8421`，比 baseline 低 `0.0731`。负向很明显。`Petrunko_3` 更像精密跟踪任务，额外动作随机性会降低按键时序和手部轨迹精度，探索收益不足。

### Value Clip

思路：启用 value function clipping，并提高 `vf_coef`，希望 critic 更稳，减少价值估计剧烈变化带来的策略更新误差。

实际作用：`best f1=0.8756`，比 baseline 低 `0.0396`。没有提升，且平台明显偏低。推测 value clipping 在 dense imitation reward 下限制了 critic 拟合速度，使 advantage 质量下降。

### gSDE

思路：使用 generalized State Dependent Exploration，让连续动作探索依赖状态，而不是固定高斯噪声。

实际作用：`best f1=0.6436`，比 baseline 低 `0.2716`，是最差的一组。当前 residual action 需要非常精确地补偿演示动作，gSDE 的探索噪声严重破坏了按键和手部控制精度。

### Dual Clip

思路：对负 advantage 样本使用 dual clipping，限制过强的负向策略更新，避免 PPO 在差样本上更新过猛。

实际作用：`best f1=0.9118`，比 baseline 低 `0.0034`；末端 `last10 mean=0.90975`，接近 baseline 但仍略低。它没有明显破坏训练，但也没有带来上限提升。

### Advantage Clip

思路：裁剪归一化后的 advantage，降低极端 advantage 样本对梯度的主导，避免少量异常轨迹影响更新。

实际作用：`best f1=0.9118`，比 baseline 低 `0.0034`。曲线接近 baseline，但没有更高平台。可能 `Petrunko_3` 的高 advantage 样本本身包含有用纠错信号，裁剪后削弱了少量关键改进。

### Large Policy

思路：扩大 policy/value 网络容量，从 `1024,256` 增加到 `2048,1024,512`，让策略表示更复杂的相位、手部姿态和按键关系。

实际作用：2000 iter 时 `best f1=0.9172`，比 baseline 高 `0.0020`，且最高点出现在最终 `iter=2000`。增加到 3000 iter 后，`large_policy_more_steps` 达到 `best f1=0.9211`，峰值出现在 `iter=2950`，同时 `best recall=0.8897`。这说明 large policy 的早期学习速度确实不如 baseline 稳定，但后期容量开始发挥作用；在 recall 实验之前，它是最值得继续观察的方向。

### Actor Large

思路：只扩大 actor 到 `2048,1024,512`，critic 保持 baseline 的 `1024,256`，目标是增加动作表达能力，同时避免 critic 过大带来的价值估计变化。

实际作用：`best f1=0.9106`，比 baseline 低 `0.0046`，也低于双塔都扩大的 `large_policy`。说明在这组参数下，只扩 actor 没有带来更高动作上限，可能 critic 容量不足反而限制了 advantage 质量。

### LayerNorm Large

思路：在 large policy 的 MLP 隐层中加入 LayerNorm，希望改善更大网络的优化条件，让容量提升更容易被训练利用。

实际作用：`best f1=0.9113`，比 baseline 低 `0.0039`，明显低于无 LayerNorm 的 `large_policy=0.9172`。LayerNorm 没有帮助，可能改变了当前 MLP/正交初始化下已经适配好的激活尺度。

### Long Horizon

思路：把 rollout horizon 从 `512` 提到 `1024`，batch size 提到 `2048`，`gae_lambda` 提到 `0.98`，希望改善长时间依赖和较远期奖励信用分配。

实际作用：该 run 现在已经完整跑到 200 个 eval 点，`best f1=0.9126`，比 baseline 低 `0.0026`，`final f1=0.9070`。它在中后期到过 0.91 以上，但最终平台不如 baseline。更长 rollout 没有提高上限，且末端有回落。

### Cosine LR

思路：使用全局 cosine learning-rate schedule，前 50 iter warmup，之后从 `3e-4` 下降到 `2e-5`，希望减少后期更新噪声并提升精修能力。

实际作用：`best f1=0.9069`，比 baseline 低 `0.0083`。曲线直到 `iter=1720` 才首次达到 0.90，明显慢于 baseline。说明当前 legacy LR 衰减方式虽然粗糙，但更适合这个任务；过低的后期学习率可能限制了继续纠错。

### Low Std

思路：降低初始 action log std，并把 log std 限制在 `[-2.5, -0.3]`，减少 residual action 采样噪声，让训练更接近精确跟踪。

实际作用：`best f1=0.9073`，比 baseline 低 `0.0079`。说明虽然 gSDE/entropy 的额外探索是负向，但把探索压得太低也会损伤性能；PPO 仍需要一定动作噪声去发现局部纠错动作。

### EMA

思路：维护 policy 参数的 exponential moving average，`decay=0.995`，评估和保存 best 时使用 EMA 参数，希望平滑后期策略抖动，提高 eval 曲线峰值。

实际作用：`best f1=0.9159`，比 baseline 高 `0.0007`，`last10 mean=0.91301` 也略高于 baseline 的 `0.91213`。这是第二批新优化中最接近有效的方向，但提升极小，不能单 seed 确认。它更像“评估平滑/模型平均”收益，而不是策略能力显著提高。

### SiLU No Ortho

思路：把激活从 GELU 改为 SiLU，关闭 orthogonal initialization，并把初始 `log_std` 设为 `-0.5`，测试不同 MLP 参数化是否能得到更好的局部最优。

实际作用：`best f1=0.9133`，比 baseline 低 `0.0019`，末端 `last10 mean=0.90940`。整体接近 baseline，但没有收益。当前 GELU + orthogonal init 的默认组合仍然更稳。

## Recall 实验前结论

在面向 recall 的第二批实验之前，`Petrunko_3` 的 baseline 已经接近原配置族的上限，但增加训练步数后，`large_policy_more_steps` 展现出更明确的后期优势：`best f1=0.9211`，`last10 mean=0.91885`，均高于 `baseline_more_steps`。这仍然是单 seed 结果，需要复跑确认，但已经比 2000 iter 的差异更有说服力。

当前指标结构也很清楚：precision 几乎到顶，recall 不到 0.9。继续只压 wrong press 或动作平滑，很可能继续提高不了 f1；下一阶段应把优化目标从“少误按”转向“少漏按”。

可保留观察的方向：

- `large_policy`：增加步数后是当时最高，建议继续做多 seed 或更长步数复跑；如果多 seed 仍高，说明容量确实是有效方向。
- `ema`：新增实验里最接近正向，成本低，可以和 `large_policy` 组合成 `large_policy + EMA`。
- `residual_reg_anneal`：比固定 residual regularization 好，几乎持平 baseline；如果还想使用动作约束，应只考虑退火版而不是固定版。

建议暂时降低优先级：

- `rsi`
- `residual_reg`
- `clip_anneal`
- `value_clip`
- `entropy_anneal`
- `gsde`
- `actor_large`
- `layer_norm_large`
- `cosine_lr`
- `low_std`
- `silu_no_ortho`
- `dual_clip`
- `advantage_clip`
- `long_horizon`
- `grpo`，除非先补一条 `Petrunko_3` 的完整曲线，否则无法在本曲目上比较。

## 面向 Recall 的新实验设计

这次新增的实现默认不改变 baseline 行为：`key_press_positive_weight=0.5`、`key_press_negative_weight=0.5`、`key_press_negative_mode=any`、`key_press_reward_scale=2.0`，仍然等价于原来的 key press reward。只有脚本显式传参时，才会切换到 recall-biased reward。

新增参数：

- `--key-press-positive-weight`：提高目标键奖励权重，鼓励按下应该按的键。
- `--key-press-negative-weight`：降低非目标键奖励权重，允许策略用少量 precision 风险换 recall。
- `--key-press-negative-mode fraction`：把 wrong-key penalty 从“任意错键即丢掉整项奖励”改为“按错比例扣分”，让探索目标键时不被单个错键过度惩罚。
- `--recall-activation-bonus-coef`：直接奖励目标键 activation，用来补强漏按音符的梯度。
- `--missed-note-penalty-coef`：直接惩罚未激活的目标键，让 missed note 更贵。
- `--checkpoint-metric recall|fbeta` 和 `--checkpoint-f-beta`：训练过程不变，但保存 checkpoint 时可以偏向 recall 或 F2，避免最终视频只选择 f1 最优但 recall 不高的点。

新增脚本：

| 脚本 | 主要思路 | 默认项目名 |
|---|---|---|
| `single_task/bash/run_ppo_recall_reward.sh` | 目标键奖励权重 `0.8`，非目标键权重 `0.2`，先测试最直接的 reward reweight | `robopianist-Petrunko_3-recall` |
| `single_task/bash/run_ppo_recall_soft_wrong.sh` | 使用 `fraction` wrong-key penalty，降低单个误按对 reward 的破坏 | `robopianist-Petrunko_3-recall` |
| `single_task/bash/run_ppo_recall_activation_bonus.sh` | 在 soft wrong penalty 基础上增加目标键 activation bonus | `robopianist-Petrunko_3-recall` |
| `single_task/bash/run_ppo_recall_missed_penalty.sh` | 在 soft wrong penalty 基础上增加 missed-note penalty | `robopianist-Petrunko_3-recall` |
| `single_task/bash/run_ppo_f2_checkpoint.sh` | baseline 训练不变，只用 F2 保存 best checkpoint，观察是否能选到更高 recall 的模型 | `robopianist-Petrunko_3-recall` |
| `single_task/bash/run_ppo_large_policy_recall.sh` | 结合 3000 iter large policy、soft wrong penalty、activation bonus 和 F2 checkpoint | `robopianist-Petrunko_3-recall` |

## 面向 Recall 的实验结果

本次 6 条实验来自 `single_task/bash/background_pids` 下 20260602_105747 的 pid 文件。实际 W&B config 里的 `project` 仍是 `robopianist-Petrunko_3`，应该是外层环境变量覆盖了脚本默认值；这里以脚本名、run name 和本地 run 目录对应关系为准。

脚本与 run 映射：

| 脚本 | pid 文件 | run |
|---|---|---|
| `run_ppo_recall_reward.sh` | `run_ppo_recall_reward_20260602_105747.pid` | `run-20260602_105823-gy66zwzt` |
| `run_ppo_recall_soft_wrong.sh` | `run_ppo_recall_soft_wrong_20260602_105747.pid` | `run-20260602_105823-vwicg3gh` |
| `run_ppo_recall_activation_bonus.sh` | `run_ppo_recall_activation_bonus_20260602_105747.pid` | `run-20260602_105823-321rka1f` |
| `run_ppo_recall_missed_penalty.sh` | `run_ppo_recall_missed_penalty_20260602_105747.pid` | `run-20260602_105823-s633rn6x` |
| `run_ppo_f2_checkpoint.sh` | `run_ppo_f2_checkpoint_20260602_105747.pid` | `run-20260602_105824-0bdqbjni` |
| `run_ppo_large_policy_recall.sh` | `run_ppo_large_policy_recall_20260602_105747.pid` | `run-20260602_105823-2rs6i77e` |

曲线指标：

| 方法 | eval 点数 | best f1 / iter | precision@best f1 | recall@best f1 | best recall / iter | best F2 / iter | final f1 | final precision | final recall | last10 f1 | last10 recall | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| recall_reward | 200 | 0.9254 / 1930 | 0.9933 | 0.8962 | 0.8962 / 1930 | 0.9141 / 1930 | 0.9247 | 0.9947 | 0.8956 | 0.92389 | 0.89431 | 第二批最高 f1，且末端平台最高 |
| recall_soft_wrong | 200 | 0.9189 / 1770 | 0.9593 | 0.9068 | 0.9068 / 1770 | 0.9168 / 1770 | 0.9109 | 0.9580 | 0.9010 | 0.91543 | 0.90306 | recall 明显提升，但 precision 掉得多，后期回落 |
| recall_activation_bonus | 200 | 0.9209 / 1990 | 0.9626 | 0.9104 | 0.9109 / 2000 | 0.9204 / 1990 | 0.9198 | 0.9599 | 0.9109 | 0.91854 | 0.90888 | activation bonus 有效拉 recall，但 f1 不如简单 reweight |
| recall_missed_penalty | 200 | 0.9220 / 1960 | 0.9606 | 0.9119 | 0.9119 / 1970 | 0.9214 / 1970 | 0.9219 | 0.9628 | 0.9113 | 0.92003 | 0.91040 | 比 activation bonus 更稳，是 2000 iter 下最好的高 recall 配置 |
| f2_checkpoint | 200 | 0.9152 / 1900 | 0.9958 | 0.8811 | 0.8812 / 2000 | 0.9020 / 2000 | 0.9144 | 0.9958 | 0.8812 | 0.91213 | 0.87835 | 只改 checkpoint 选择几乎不改变结果 |
| large_policy_recall | 300 | 0.9231 / 2970 | 0.9570 | 0.9171 | 0.9190 / 2880 | 0.9258 / 2880 | 0.9215 | 0.9560 | 0.9164 | 0.92007 | 0.91592 | 第二批 f1 次高，但 recall 和 F2 最高 |

相对上一轮最佳 `large_policy_more_steps`：`large_policy_more_steps` 的 `best f1=0.9211`、`precision@best=0.9985`、`recall@best=0.8897`。本轮 `recall_reward` 把 f1 提到 `0.9254`，主要来自 recall 提升到 `0.8962`，precision 仍保持 `0.9933`；`large_policy_recall` 的 f1 只有 `0.9231`，但 recall 达到 `0.9171`，说明它确实按下了更多目标键，只是付出了更明显的 precision 代价。

### Recall Reward

思路：把目标键奖励权重提高到 `0.8`，非目标键权重降到 `0.2`，但仍使用 `any` wrong-key penalty。它不是完全放开误按，而是在原有严格 wrong-key 逻辑下增加目标键的相对收益。

实际作用：`best f1=0.9254`，比 baseline 高 `0.0102`，比上一轮最佳 `large_policy_more_steps` 高 `0.0043`。峰值处 `precision=0.9933`、`recall=0.8962`，precision 仍接近 1，recall 比原 baseline 的 `0.8811` 和 `large_policy_more_steps` 的 `0.8897` 都更高。结论是：当前最有效的方向不是彻底软化 wrong-key 惩罚，而是先做温和的正负 key reward reweight。

### Recall Soft Wrong

思路：使用 `fraction` wrong-key penalty，目标键/非目标键权重为 `0.7/0.3`，让单个错键不会直接清空负项奖励。

实际作用：`best f1=0.9189`，虽然高于 baseline，但低于 `large_policy_more_steps` 和本轮其他 recall 配置。它把 `recall` 提到 `0.9068`，但 `precision` 降到 `0.9593`，末端 `final f1=0.9109` 明显回落。结论是：soft wrong penalty 确实能让策略更愿意按目标键，但单独使用时 precision 代价过大，训练后期也不够稳。

### Recall Activation Bonus

思路：在 `fraction` wrong-key penalty 上加 `recall_activation_bonus_coef=0.2`，直接奖励目标键 activation。

实际作用：`best f1=0.9209`，`best recall=0.9109`，比 `recall_soft_wrong` 的 f1 和末端稳定性都更好。它证明直接给目标键 activation 梯度是有效的，但峰值 `precision=0.9626`，precision 损失仍然明显，因此 f1 低于 `recall_reward`。

### Recall Missed Penalty

思路：在 `fraction` wrong-key penalty 上加 `missed_note_penalty_coef=0.2`，直接惩罚未激活的目标键。

实际作用：`best f1=0.9220`，`best recall=0.9119`，`final f1=0.9219`，last10 f1 也达到 `0.92003`。它比 activation bonus 稍强，说明“惩罚漏按”比“奖励激活”在这组参数下更稳。作为 2000 iter 的高 recall 配置，它是最值得继续微调的一条。

### F2 Checkpoint

思路：训练 reward 完全保持 baseline，只把 checkpoint 选择从 f1 改成 F2。

实际作用：曲线几乎复现 baseline，`best f1=0.9152`，`best F2=0.9020` 出现在最后 `iter=2000`。它能在同一条曲线里选择稍高 recall 的 checkpoint，但幅度很小，不能替代训练 reward 的 recall bias。结论是：F2 checkpoint 只适合作为配套选择规则，不是独立优化手段。

### Large Policy Recall

思路：把上一轮有效的 3000 iter large policy 和 recall-biased reward 组合起来，使用 `2048,1024,512` actor/critic，`0.75/0.25` 正负 key 权重，`fraction` wrong-key penalty，`recall_activation_bonus_coef=0.15`，并用 F2 保存 best checkpoint。

实际作用：`best f1=0.9231`，低于 `recall_reward=0.9254`，但它给出了当前最高 `best recall=0.9190` 和 `best F2=0.9258`。F2 最优点在 `iter=2880`，f1 最优点在 `iter=2970`，说明 F2 checkpoint 确实会选择更偏 recall 的模型。按 2026-06-02 的结果看，如果目标是 f1，优先复跑 `recall_reward`；如果目标是减少漏按或用 F2 做选择，`large_policy_recall` 是当时最强方向。

## 2026-06-02 更新后的当前结论

面向 recall 的 reward 调整是当时最有效的一组改动。`recall_reward` 以最小结构改动拿到当时最高 `best f1=0.9254`，而且 precision 仍保持在 `0.99+`，说明之前的判断是正确的：瓶颈主要是漏按，不是误按。

`fraction` wrong-key penalty 让 recall 大幅提高，但 precision 会明显下降。它需要 activation bonus、missed penalty 或 large policy 配合，不能单独作为默认配置。2000 iter 下，`recall_missed_penalty` 比 `recall_activation_bonus` 更稳；3000 iter 下，`large_policy_recall` 的 recall/F2 最强，但 f1 仍不如简单的 `recall_reward`。

下一步建议优先级：

- 第一优先级：复跑 `recall_reward` 多 seed，确认 `best f1=0.9254` 是否稳定。
- 第二优先级：在 `recall_reward` 基础上小范围扫 `positive/negative` 权重，例如 `0.75/0.25`、`0.85/0.15`，先不要切到 `fraction`。
- 第三优先级：复跑或微调 `recall_missed_penalty`，重点看能否在 `recall>0.91` 的同时把 precision 拉回到 `0.97+`。
- 第四优先级：保留 `large_policy_recall` 作为 F2/高 recall 方向，适合减少漏按，但如果最终评估仍看 f1，它不是当前首选。
- 低优先级：`f2_checkpoint` 单独使用收益很小；`recall_soft_wrong` 单独使用不稳定，不建议继续单独扩展。

## 2026-06-03 Recall 参数扫描结果

本次 7 条实验来自 `single_task/bash/background_pids` 下 20260603_010643 的 pid 文件，全部使用 `total_iters=2000` 和默认 W&B project `robopianist-Petrunko_3`。这些实验分别测试 `recall_reward` 的正负权重扫描、`recall_missed_penalty` 的 precision recovery 微调，以及 `large_policy + recall_reward` 的组合。

脚本与 run 映射：

| 脚本 | pid 文件 | run |
|---|---|---|
| `run_ppo_recall_reward_p075_n025.sh` | `run_ppo_recall_reward_p075_n025_20260603_010643.pid` | `run-20260603_010720-bab4h8ta` |
| `run_ppo_recall_reward_p085_n015.sh` | `run_ppo_recall_reward_p085_n015_20260603_010643.pid` | `run-20260603_010721-n7ttogah` |
| `run_ppo_recall_missed_penalty_p070_n030_m010.sh` | `run_ppo_recall_missed_penalty_p070_n030_m010_20260603_010643.pid` | `run-20260603_010720-6eqbetk7` |
| `run_ppo_recall_missed_penalty_p065_n035_m010.sh` | `run_ppo_recall_missed_penalty_p065_n035_m010_20260603_010643.pid` | `run-20260603_010720-caycj3hf` |
| `run_ppo_recall_missed_penalty_p065_n035_m015.sh` | `run_ppo_recall_missed_penalty_p065_n035_m015_20260603_010643.pid` | `run-20260603_010720-pns0dede` |
| `run_ppo_recall_missed_penalty_p060_n040_m015.sh` | `run_ppo_recall_missed_penalty_p060_n040_m015_20260603_010643.pid` | `run-20260603_010721-q0vv8v3w` |
| `run_ppo_large_policy_recall_reward.sh` | `run_ppo_large_policy_recall_reward_20260603_010643.pid` | `run-20260603_010720-zobk1vmo` |

曲线指标：

| 方法 | eval 点数 | best f1 / iter | precision@best f1 | recall@best f1 | best recall / iter | best F2 / iter | final f1 | final precision | final recall | last10 f1 | last10 recall | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| recall_reward_p075_n025 | 200 | 0.9250 / 1850 | 0.9976 | 0.8942 | 0.8942 / 1850 | 0.9131 / 1850 | 0.9248 | 0.9980 | 0.8934 | 0.92270 | 0.89141 | 接近 0.8/0.2，但 recall 没有继续抬高 |
| recall_reward_p085_n015 | 200 | 0.9322 / 1920 | 0.9930 | 0.9050 | 0.9053 / 1990 | 0.9214 / 1990 | 0.9289 | 0.9928 | 0.9023 | 0.92974 | 0.90383 | 本轮最高 f1，strict wrong-key 下继续偏 recall 有效 |
| recall_missed_penalty_p070_n030_m010 | 200 | 0.9197 / 1760 | 0.9681 | 0.9054 | 0.9073 / 2000 | 0.9185 / 2000 | 0.9183 | 0.9663 | 0.9073 | 0.91541 | 0.90300 | 降低 missed penalty 后 recall 和 f1 都低于原版 |
| recall_missed_penalty_p065_n035_m010 | 200 | 0.9103 / 1970 | 0.9553 | 0.8991 | 0.9014 / 1950 | 0.9105 / 1950 | 0.9079 | 0.9541 | 0.8985 | 0.90704 | 0.89856 | 明显负向，precision 没拉回，recall 也下降 |
| recall_missed_penalty_p065_n035_m015 | 200 | 0.9191 / 1980 | 0.9561 | 0.9083 | 0.9086 / 1950 | 0.9179 / 1950 | 0.9165 | 0.9564 | 0.9054 | 0.91586 | 0.90560 | 比 0.10 penalty 好，但仍低于原版和目标 |
| recall_missed_penalty_p060_n040_m015 | 200 | 0.9246 / 1690 | 0.9703 | 0.9085 | 0.9121 / 1920 | 0.9218 / 1920 | 0.9200 | 0.9615 | 0.9095 | 0.92104 | 0.91002 | 微调组最好，但未同时满足 recall>0.91 和 precision>0.97 |
| large_policy_recall_reward | 200 | 0.9317 / 1920 | 0.9971 | 0.9026 | 0.9026 / 1920 | 0.9200 / 1920 | 0.9283 | 0.9971 | 0.8989 | 0.92833 | 0.89929 | large policy 与 recall_reward 可叠加，f1 接近本轮最高 |

### Recall Reward Weight Sweep

本轮把已经跑过的 `0.8/0.2` 作为参照，只新增 `0.75/0.25` 和 `0.85/0.15`。结果显示，`0.75/0.25` 的 `best f1=0.9250`，几乎等同于 `0.8/0.2` 的 `0.9254`，但 recall 更低；`0.85/0.15` 则显著提升到 `best f1=0.9322`，峰值处 `precision=0.9930`、`recall=0.9050`。

结论：在保留 `key_press_negative_mode=any` 的前提下，继续提高 target-key 正项权重没有造成明显 precision 崩坏，反而把 recall 推过了 `0.90`。这说明当前最有效的 recall 优化仍是 strict wrong-key penalty 下的 reward reweight，而不是先软化 wrong-key 惩罚。

### Recall Missed Penalty Tune

这组实验的目标是微调 `recall_missed_penalty`，看能否在 `recall>0.91` 的同时把 precision 拉回 `0.97+`。结果没有同时满足目标。最接近的是 `p060_n040_m015`：best f1 点 `precision=0.9703`，但 `recall=0.9085`；best recall 点达到 `recall=0.9121`，但 precision 降到 `0.9630`。

对比原始 `recall_missed_penalty` 的 `best f1=0.9220`、`precision=0.9606`、`recall=0.9119`，`p060_n040_m015` 把 f1 提到 `0.9246`，并在 best f1 点恢复到 `precision=0.9703`，但 recall 没有守住 `0.91`。其他三组提高 negative weight 或降低 missed penalty 的配置都低于原版，说明这条路对权重较敏感，继续盲目加 negative weight 不可靠。

### Large Policy Recall Reward

`large_policy_recall_reward` 使用 `2048,1024,512` actor/critic，并采用 `recall_reward` 的 `0.8/0.2 + any`。它达到 `best f1=0.9317`，峰值处 `precision=0.9971`、`recall=0.9026`，明显超过之前的 `large_policy=0.9172`、`large_policy_more_steps=0.9211` 和第二批 `large_policy_recall=0.9231`。

结论：large policy 与 strict wrong-key 的 recall reward reweight 是可叠加的。它没有像 `large_policy_recall` 那样牺牲大量 precision，而是在 precision 接近 1 的情况下把 recall 拉到 `0.90+`。不过它仍略低于小网络的 `recall_reward_p085_n015=0.9322`，说明当前收益主要来自 reward 权重，而不是单纯网络容量。

## 2026-06-03 010643 扫描结论

在 010643 这一轮扫描中，f1 最强配置是 `recall_reward_p085_n015`，`best f1=0.9322`，比 baseline 高 `0.0170`，比原 `recall_reward=0.9254` 高 `0.0068`。这条结果把 recall 提到 `0.9050`，同时 precision 仍有 `0.9930`，是当轮最干净的提升。

同轮第二强是 `large_policy_recall_reward`，`best f1=0.9317`，precision 更高但 recall 略低。它证明 large policy 适合与 strict `recall_reward` 组合，而不适合优先与 `fraction` wrong-key penalty 组合。

`recall_missed_penalty` 微调没有达成“`recall>0.91` 且 `precision>0.97`”的目标。`p060_n040_m015` 是这组里最好的折中，但它仍不如 `recall_reward_p085_n015`，因此 missed-penalty 方向应降级为备用。

基于这轮结果的后续方向：

- 优先多 seed 复跑 `recall_reward_p085_n015`，确认 `0.9322` 是否稳定。
- 做 `recall_reward` 更高正项权重的小扫，例如 `0.90/0.10`，但仍保持 `key_press_negative_mode=any`。
- 组合 `large_policy + recall_reward_p085_n015`，并考虑 3000 iter 版本，验证 large policy 是否能在更优 reward 权重下继续受益。
- 暂停 `recall_missed_penalty_p065_*` 系列；如果继续 missed-penalty，只保留 `p060_n040_m015` 附近做很小范围搜索。

## 2026-06-03 110302 Recall Reward 高正项权重扫描结果

本次 3 条实验来自 `single_task/bash/background_pids` 下 20260603_110302 的 pid 文件，全部使用 `total_iters=2000`、`seed=42`、`key_press_negative_mode=any` 和默认 W&B project `robopianist-Petrunko_3`。这组实验继续沿着 strict wrong-key penalty 下的 `recall_reward` 权重方向，把正项权重提高到 `0.90`、`0.95` 和 `0.99`。

脚本与 run 映射：

| 脚本 | pid 文件 | run |
|---|---|---|
| `run_ppo_recall_reward_p090_n010.sh` | `run_ppo_recall_reward_p090_n010_20260603_110302.pid` | `run-20260603_110330-2akh2erl` |
| `run_ppo_recall_reward_p095_n005.sh` | `run_ppo_recall_reward_p095_n005_20260603_110302.pid` | `run-20260603_110330-lds3pxhv` |
| `run_ppo_recall_reward_p099_n001.sh` | `run_ppo_recall_reward_p099_n001_20260603_110302.pid` | `run-20260603_110330-bo4j6qbr` |

曲线指标：

| 方法 | eval 点数 | best f1 / iter | precision@best f1 | recall@best f1 | best recall / iter | best F2 / iter | final f1 | final precision | final recall | last10 f1 | last10 recall | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| recall_reward_p090_n010 | 200 | 0.9327 / 1930 | 0.9928 | 0.9061 | 0.9069 / 1980 | 0.9227 / 1760 | 0.9312 | 0.9914 | 0.9063 | 0.92936 | 0.90427 | 高于 0.85/0.15，末端保持在 0.93 附近 |
| recall_reward_p095_n005 | 200 | 0.9331 / 1700 | 0.9881 | 0.9081 | 0.9090 / 1890 | 0.9234 / 1580 | 0.9320 | 0.9868 | 0.9072 | 0.92964 | 0.90670 | 当前最高 f1，recall 继续提高且 precision 仍可接受 |
| recall_reward_p099_n001 | 200 | 0.9269 / 1950 | 0.9684 | 0.9124 | 0.9124 / 1950 | 0.9231 / 1950 | 0.9232 | 0.9666 | 0.9099 | 0.92387 | 0.91072 | recall 最高，但 precision 代价过大，f1 明显回落 |

### 高正项权重结论

`recall_reward_p095_n005` 达到新的最高 `best f1=0.9331`，比 baseline 高 `0.0179`，比 `recall_reward_p085_n015=0.9322` 高 `0.0009`。提升幅度不大，但它把 best f1 点的 recall 从 `0.9050` 继续推到 `0.9081`，precision 仍有 `0.9881`，说明在 strict wrong-key penalty 下继续提高目标键权重仍然有效。

`recall_reward_p090_n010` 的表现非常接近，`best f1=0.9327`、`precision=0.9928`、`recall=0.9061`，比 `0.95/0.05` 更保守，precision 更高、recall 略低。结合 last10 指标，`0.90/0.10` 和 `0.95/0.05` 都值得多 seed 复跑。

`recall_reward_p099_n001` 证明权重不能无限偏向目标键：它把 best recall 提到 `0.9124`，但 precision 降到 `0.9684`，导致 best f1 只有 `0.9269`，低于 `0.85/0.15`、`0.90/0.10` 和 `0.95/0.05`。如果最终目标是 f1，`0.99/0.01` 不适合作为默认配置；如果目标是减少漏按或提高 F2，它可以作为高 recall 参考点。

## 2026-06-04 Petrunko 当前结论

当前 f1 最强配置是 `recall_reward_p095_n005`，`best f1=0.9331`，但它只比 `recall_reward_p090_n010=0.9327` 和 `recall_reward_p085_n015=0.9322` 略高。考虑单 seed 波动，不能仅凭这一次差距断定 `0.95/0.05` 稳定优于 `0.90/0.10` 或 `0.85/0.15`；比较确定的是最佳区间已经从 `0.80/0.20` 上移到 `0.90/0.10` 到 `0.95/0.05` 附近。

`0.99/0.01` 已经越过 f1 最优点：recall 继续上升，但 precision 损失开始主导，f1 回落明显。因此后续不建议继续把 negative weight 压到接近 0，除非评估目标改成 recall/F2。

下一步建议：

- 优先多 seed 复跑 `recall_reward_p095_n005` 和 `recall_reward_p090_n010`，确认 `0.9331` 和 `0.9327` 的差异是否稳定。
- 在 `0.90/0.10` 到 `0.95/0.05` 之间做更小步长扫描，例如 `0.92/0.08`、`0.94/0.06`、`0.96/0.04`。
- 组合 `large_policy + recall_reward_p090_n010` 或 `large_policy + recall_reward_p095_n005`，并考虑 3000 iter，验证 large policy 在更优 reward 权重下是否继续受益。
- `recall_reward_p099_n001` 只保留为高 recall/F2 参考，不建议作为 f1 默认方向。

## 2026-06-05 跨曲目 Large Policy + Recall Reward p095/n005 结果

本次 3 条实验来自 `single_task/bash/background_pids` 下 20260604_113826 的 pid 文件，全部使用 `seed=42`、`total_iters=2000`、`2048,1024,512` actor/critic、`key_press_positive_weight=0.95`、`key_press_negative_weight=0.05` 和 `key_press_negative_mode=any`。三条 run 都完整跑到 `iter=2000/2000`。

脚本与 run 映射：

| 曲目 | 脚本 | pid 文件 | run |
|---|---|---|---|
| `Pirates_2` | `run_ppo_large_policy_recall_reward_p095_n005_pirates_2.sh` | `run_ppo_large_policy_recall_reward_p095_n005_pirates_2_20260604_113826.pid` | `run-20260604_113858-7sq3tzyq` |
| `Stan_1` | `run_ppo_large_policy_recall_reward_p095_n005_stan_1.sh` | `run_ppo_large_policy_recall_reward_p095_n005_stan_1_20260604_113826.pid` | `run-20260604_113858-epae2sz0` |
| `Petrunko_3` | `run_ppo_large_policy_recall_reward_p095_n005_petrunko_3.sh` | `run_ppo_large_policy_recall_reward_p095_n005_petrunko_3_20260604_113826.pid` | `run-20260604_113859-ulafj8kx` |

曲线指标：

| 曲目 | eval 点数 | best f1 / iter | precision@best f1 | recall@best f1 | best recall / iter | best F2 / iter | final f1 | final precision | final recall | last10 f1 | last10 recall | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Pirates_2 | 200 | 0.9727 / 2000 | 0.9933 | 0.9672 | 0.9683 / 1890 | 0.9729 / 1770 | 0.9727 | 0.9933 | 0.9672 | 0.97182 | 0.96745 | best f1 出现在最后，末端稳定，略高于本地历史最好 |
| Stan_1 | 200 | 0.9891 / 1990 | 0.9980 | 0.9880 | 0.9885 / 1950 | 0.9903 / 1950 | 0.9887 | 0.9984 | 0.9867 | 0.98884 | 0.98764 | 明显高于本地 baseline，precision 基本保持到顶 |
| Petrunko_3 | 200 | 0.9311 / 1900 | 0.9879 | 0.9079 | 0.9085 / 1740 | 0.9228 / 1900 | 0.9280 | 0.9851 | 0.9064 | 0.92708 | 0.90474 | 不如小网络 0.95/0.05，也略低于 large_policy + 0.8/0.2 |

与本地参考 run 的比较：

| 曲目 | 参考配置 | 参考 run | 参考 best f1 | 本次 best f1 | delta | 说明 |
|---|---|---|---:|---:|---:|---|
| Pirates_2 | baseline | `run-20260527_212310-zqoeamxu` | 0.9698 | 0.9727 | +0.0029 | 比 baseline 高，recall 从 `0.9631` 提到 `0.9672`，precision 从 `0.9967` 降到 `0.9933` |
| Pirates_2 | residual_reg | `run-20260528_165346-bz3xef7e` | 0.9713 | 0.9727 | +0.0014 | 也略高于之前本地最高的 residual_reg |
| Stan_1 | baseline | `run-20260521_165236-ssa832y8` | 0.9857 | 0.9891 | +0.0034 | recall 从 `0.9805` 提到 `0.9880`，precision 仍接近 1 |
| Petrunko_3 | recall_reward_p095_n005 | `run-20260603_110330-lds3pxhv` | 0.9331 | 0.9311 | -0.0020 | 加 large policy 后没有超过小网络同权重 |
| Petrunko_3 | large_policy_recall_reward | `run-20260603_010720-zobk1vmo` | 0.9317 | 0.9311 | -0.0006 | `0.95/0.05` 不如 large policy 的 `0.8/0.2` 版本 |

### 跨曲目结论

`large_policy + recall_reward_p095_n005` 在 `Pirates_2` 和 `Stan_1` 上是正向的：两首歌都主要通过提高 recall 获得 f1 提升，precision 仍保持在 `0.99+` 或接近 `1.0`。其中 `Stan_1` 的收益更明确，`best f1=0.9891`，比本地 baseline 高 `0.0034`。

`Petrunko_3` 上的组合结果没有继续提高上限：`best f1=0.9311`，低于小网络 `recall_reward_p095_n005=0.9331`，也略低于 `large_policy_recall_reward=0.9317`。这说明对 `Petrunko_3` 而言，`0.95/0.05` 的最优性目前更像 reward 权重本身的收益，并没有和 large policy 形成稳定叠加。

后续建议：

- `Stan_1` 和 `Pirates_2` 可以继续用 `large_policy + recall_reward_p095_n005` 做多 seed 或更长步数复跑，确认小幅收益是否稳定。
- `Petrunko_3` 不建议优先扩展 `large_policy + 0.95/0.05`；若继续 large policy，优先测试 `0.90/0.10` 或 3000 iter，而不是继续提高正项权重。
- 跨曲目结论支持 recall reweight 的泛化性，但不同曲目的最优网络容量和正负权重不完全一致。
