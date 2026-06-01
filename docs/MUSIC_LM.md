# MAESTRO Music LM and PPL Reward

This workflow trains a small GPT-style next-token model on MAESTRO MIDI and uses
its perplexity as an evaluation metric or optional PPO auxiliary reward.

## Feasibility

This is feasible as an auxiliary signal, not as a replacement for PianoMime's
target-song F1 metrics. MAESTRO is piano-only MIDI performance data, so the
domain matches RoboPianist's generated MIDI events well. PPL can catch musical
artifacts that note F1 misses, such as ghost notes, odd local ordering, unnatural
rhythm, or strange sustain behavior.

The main caveat is objective mismatch: low PPL means "looks like the MAESTRO
training distribution", not "matches this target MIDI." Use it first as an
evaluation metric, then as a small bounded reward term while keeping the original
note-level reward.

## Prepare MAESTRO

Create the dedicated virtualenv first:

```bash
bash scripts/setup_music_lm_env.sh
```

```bash
bash scripts/prepare_music_lm_data.sh
```

To force a fresh download:

```bash
DOWNLOAD=1 bash scripts/prepare_music_lm_data.sh
```

For a quick smoke test:

```bash
LIMIT_PER_SPLIT=2 bash scripts/prepare_music_lm_data.sh
```

The tokenizer uses time-shift, velocity, note-on, note-off, and sustain events.
Durations are represented by note-on/note-off distance.

## Train

```bash
bash scripts/train_music_lm.sh
```

Override training length or device when needed:

```bash
MAX_STEPS=1000 DEVICE=cpu bash scripts/train_music_lm.sh
```

The trainer writes `best.pt`, `latest.pt`, and `train_log.jsonl`.

## A100 Launch

On an A100/CUDA node, use the dedicated launcher:

```bash
bash scripts/train_music_lm_a100.sh
```

The launcher checks that CUDA is available, enables CUDA training through
`DEVICE=cuda`, and uses the Music LM virtualenv from `configs/baseline.toml`.
If the virtualenv is missing, it creates it automatically.

If `artifacts/maestro_tokens/train.bin` is already present, training starts
immediately. If the node does not have tokenized MAESTRO yet, either copy
`artifacts/maestro_tokens` onto the node or let the launcher download and
tokenize MAESTRO:

```bash
A100_DOWNLOAD_MAESTRO=1 bash scripts/train_music_lm_a100.sh
```

Useful overrides:

```bash
MAX_STEPS=1000 bash scripts/train_music_lm_a100.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_music_lm_a100.sh
```

The default model has 3,370,752 trainable parameters. Raw fp32 weights are about
13.5 MB. The current checkpoint format stores model weights plus small config
metadata, not optimizer state, so `best.pt` / `latest.pt` are expected to be
about 16-18 MB each. Keeping both files costs roughly 32-36 MB.

## Current Trained Checkpoint

The repository includes the A100-trained checkpoint at:

```text
artifacts/music_lm/small_gpt/best.pt
```

Training summary:

```text
step: 19000
best_validation_loss: 1.784455587863922
checkpoint_size: 16.88 MiB
sha256: 0e88246634ebf525271e76d89fdbe47cfb9b58d9eee4756fe748858917749d8e
```

Smoke-test evaluation on `tutorial/Stan_1.mid`:

```text
tokens: 914
log_ppl: 4.2667458274147725
ppl: 71.2892701570504
```

## Evaluate MIDI

```bash
bash scripts/eval_music_lm.sh tutorial/Stan_1.mid
```

The output reports token count, `log_ppl`, and `ppl`.

## PPO Auxiliary Reward

`single_task/train_ppo.py` now accepts optional music-LM reward arguments:

```bash
bash scripts/run_ppo_with_music_lm.sh Petrunko_3
```

The reward bonus is:

```text
bonus = -weight * (window_log_ppl - reference_log_ppl)
```

Prefer setting `reference_log_ppl` from the target MIDI or a baseline rollout
once you have measured it.
