# MAESTRO Music LM 与 PPL Reward

本文档说明如何把 MAESTRO 钢琴 MIDI 预训练成一个小 GPT，并把得到的
perplexity 作为 PianoMime/RoboPianist 的 evaluation metric 或 PPO 辅助 reward。

## 可行性判断

这个方向可行，但建议先作为 evaluation metric 使用，再作为低权重 reward 做消融。

优点：

- MAESTRO 是纯钢琴演奏数据，和 RoboPianist/PianoMime 的输出域匹配。
- MIDI-only 版本足够训练 next-token prediction，不需要处理音频。
- PPL 能补充 F1 看不到的错误，例如 ghost notes、局部乱序、奇怪节奏和不自然 sustain。
- 当前仓库的仿真环境已经能通过 `piano.midi_module` 记录机器人实际触发的 MIDI events，因此可以直接 score 机器人演奏。

主要风险：

- PPL 衡量“像 MAESTRO 训练分布”，不等价于“弹对目标曲子”。它不能替代目标 MIDI 的 precision/recall/F1。
- 作为 reward 时，模型可能偏好常见古典片段而不是当前目标曲。建议 reward 写成辅助项，并保留原始 note-level reward。
- 如果只用全局 PPL，训练信号会很稀疏。当前实现用 sliding-window log-PPL，在机器人产生新 MIDI event 时才加一个有界 bonus。
- MAESTRO 授权是非商业共享协议；课程研究通常没问题，公开发布模型时要标注数据来源和许可。

## 数据准备

本仓库已经提供脚本化入口。先创建专用 virtualenv：

```bash
bash scripts/setup_music_lm_env.sh
```

下载并 tokenize MAESTRO v3.0.0 MIDI-only：

```bash
bash scripts/prepare_music_lm_data.sh
```

如果需要重新下载，使用：

```bash
DOWNLOAD=1 bash scripts/prepare_music_lm_data.sh
```

默认路径来自 `configs/baseline.toml` 的 `[music_lm]` 段：

```text
artifacts/maestro
artifacts/maestro_tokens
artifacts/music_lm/small_gpt
```

快速 smoke test 可以限制每个 split 的曲目数：

```bash
LIMIT_PER_SPLIT=2 bash scripts/prepare_music_lm_data.sh
```

tokenizer 是事件流格式：

- `TIME_SHIFT_k`
- `VELOCITY_k`
- `NOTE_ON_pitch`
- `NOTE_OFF_pitch`
- `SUSTAIN_ON`
- `SUSTAIN_OFF`

默认时间量化为 10 ms，最大单 token time shift 为 1 s，超出时重复多个
time-shift token。

## 训练小 GPT

```bash
bash scripts/train_music_lm.sh
```

临时覆盖训练步数或设备：

```bash
MAX_STEPS=1000 DEVICE=cpu bash scripts/train_music_lm.sh
```

产物：

```text
artifacts/music_lm/small_gpt/
  best.pt
  latest.pt
  train_log.jsonl
```

`best.pt` 会保存模型权重、GPT config 和 tokenizer config。

## A100 一键启动

在 A100/CUDA 节点上使用专用启动脚本：

```bash
bash scripts/train_music_lm_a100.sh
```

这个脚本会检查 CUDA 是否可用，把训练设备设为 `DEVICE=cuda`，并使用
`configs/baseline.toml` 里配置的 Music LM virtualenv。如果 virtualenv 不存在，
脚本会自动创建并安装依赖。

如果 `artifacts/maestro_tokens/train.bin` 已经存在，脚本会直接开始训练。如果
A100 节点还没有 tokenized MAESTRO，可以先把本地的 `artifacts/maestro_tokens`
复制过去，或者让脚本自动下载并 tokenize：

```bash
A100_DOWNLOAD_MAESTRO=1 bash scripts/train_music_lm_a100.sh
```

常用覆盖参数：

```bash
MAX_STEPS=1000 bash scripts/train_music_lm_a100.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_music_lm_a100.sh
```

默认模型参数量是 3,370,752。纯 fp32 权重大约 13.5 MB。当前 checkpoint 只保存
模型权重和少量 config，不保存 optimizer state，所以 `best.pt` / `latest.pt`
预计各约 16-18 MB；两个都保留大约占 32-36 MB。

## 当前已训练 checkpoint

仓库已包含 A100 训练产物：

```text
artifacts/music_lm/small_gpt/best.pt
```

训练摘要：

```text
step: 19000
best_validation_loss: 1.784455587863922
checkpoint_size: 16.88 MiB
sha256: 0e88246634ebf525271e76d89fdbe47cfb9b58d9eee4756fe748858917749d8e
```

在 `tutorial/Stan_1.mid` 上的 smoke-test evaluation：

```text
tokens: 914
log_ppl: 4.2667458274147725
ppl: 71.2892701570504
```

## 作为 evaluation metric

对任意 MIDI 文件打分：

```bash
bash scripts/eval_music_lm.sh tutorial/Stan_1.mid
```

输出包含：

- `tokens`
- `log_ppl`
- `ppl`

建议报告 `log_ppl`，因为它比原始 PPL 更稳定，也更适合做 reward。

## 作为 PPO 辅助 reward

`single_task/train_ppo.py` 已加入可选参数，默认不开启。启用示例：

```bash
bash scripts/run_ppo_with_music_lm.sh Petrunko_3
```

wrapper 的 reward 形式是：

```text
bonus = -weight * (window_log_ppl - reference_log_ppl)
```

如果没有设置 `reference_log_ppl`，默认中心值为 0；实际实验中更推荐先用
目标 MIDI 的 `log_ppl` 或 baseline rollout 的 `log_ppl` 作为 reference，让 bonus
只奖励“比参考更自然”的局部片段。

## Baseline 对比口径

真正有意义的控制实验不是“短训 LM vs A100 LM”，而是同一首歌、同一个 seed、
同一套 PPO 配置和同样训练预算下的 PPO A/B：

```bash
python scripts/run_ppo_from_config.py Petrunko_3 \
  --run-name Petrunko_3_no_music_lm_seed42

bash scripts/run_ppo_with_music_lm.sh Petrunko_3 \
  Petrunko_3_with_music_lm_seed42
```

两个 run 都会写 `eval_metrics.csv`。不加 LM 的 run 记录 note/sustain 指标；
加 Music LM 的 run 记录相同指标，并额外记录：

```text
music_lm_log_ppl
music_lm_ppl
```

成功标准应该是：F1 提升或至少不下降，同时 Music LM PPL 下降。只有 PPL 降低
但 F1 下降，不能算控制效果提升。

当前本地验证只能证明训练好的 checkpoint 能提供有效 reward/evaluation signal。
在 `tutorial/Stan_1.mid` 上，A100 checkpoint 对干净 MIDI 和破坏 token stream
的区分如下：

```text
clean MIDI ppl: 71.29
same tokens shuffled ppl: 1707.07
random tokens ppl: 14165.72
```

这说明它适合作为辅助信号，但“策略确实提升”的结论仍需要上面的 PPO A/B 实验。

## 推荐实验顺序

1. 只训练 music LM，并在 MAESTRO validation/test 上确认 validation PPL 下降。
2. 对目标 MIDI、single-song replay 生成的 MIDI、失败 rollout MIDI 分别打分，检查 PPL 是否符合直觉。
3. 把 PPL 作为 evaluation metric 加入结果表，不改变 PPO reward。
4. 以很小权重加入 PPO reward，例如 `0.001`、`0.003`、`0.01`，和原 baseline 对比 F1 与 PPL。
5. 如果 F1 下降但 PPL 上升，说明模型在奖励“像古典音乐”而不是“弹对目标曲”，需要降低权重或改成只惩罚明显异常 event。
