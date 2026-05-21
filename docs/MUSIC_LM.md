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
