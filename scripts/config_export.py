#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pianomime_config import load_config, section, shell_default


def emit_paths(cfg: dict) -> list[str]:
    paths = section(cfg, "paths")
    return [
        shell_default("SHARED_ROOT", paths.get("shared_root")),
        shell_default("PROJECT_DIR", paths.get("project_dir")),
        shell_default("RUNTIME_ROOT", paths.get("runtime_root")),
        shell_default("RUNTIME_DIR", paths.get("runtime_dir")),
        shell_default("VENV", paths.get("venv")),
        shell_default("PYTHON_BIN", paths.get("python_bin")),
        shell_default("RESULTS_DIR", paths.get("results_dir")),
        shell_default("LOCAL_RESULTS_DIR", paths.get("local_results_dir")),
        shell_default("ARTIFACT_CACHE", paths.get("artifact_cache")),
    ]


def emit_environment(cfg: dict) -> list[str]:
    env = section(cfg, "environment")
    return [
        shell_default("MUJOCO_GL", env.get("mujoco_gl")),
        shell_default("XLA_PYTHON_CLIENT_PREALLOCATE", env.get("xla_python_client_preallocate")),
        shell_default("WANDB_MODE", env.get("wandb_mode")),
    ]


def emit_scheduler(cfg: dict) -> list[str]:
    sched = section(cfg, "scheduler")
    single = section(cfg, "single_song")
    multisong = section(cfg, "multisong")
    lines = [
        shell_default("SESSION", sched.get("session")),
        shell_default("RUN_ID", sched.get("run_id")),
        shell_default("GPU_IDS", sched.get("gpu_ids")),
        shell_default("GPU_FREE_MEM_MB", sched.get("gpu_free_mem_mb")),
        shell_default("POLL_SECONDS", sched.get("poll_seconds")),
        shell_default("SKIP_RUNTIME_SYNC", sched.get("skip_runtime_sync")),
        shell_default("SINGLE_REPLAY_TASKS", single.get("replay_songs")),
        shell_default("MULTISONG_TASKS", multisong.get("songs")),
        shell_default("PPO_TASKS", single.get("ppo_songs")),
    ]
    return lines


def emit_artifacts(cfg: dict) -> list[str]:
    artifacts = section(cfg, "artifacts")
    paths = section(cfg, "paths")
    cache = paths.get("artifact_cache")
    return [
        shell_default("DATASET_ID", artifacts.get("dataset_id")),
        shell_default("CHECKPOINT_ID", artifacts.get("checkpoint_id")),
        shell_default("DATASET_ZIP", f"{cache}/{artifacts.get('dataset_zip_name')}" if cache else None),
        shell_default("CHECKPOINT_ZIP", f"{cache}/{artifacts.get('checkpoint_zip_name')}" if cache else None),
    ]


def emit_music_lm(cfg: dict) -> list[str]:
    music_lm = section(cfg, "music_lm")
    return [
        shell_default("MUSIC_LM_MAESTRO_ROOT", music_lm.get("maestro_root")),
        shell_default("MUSIC_LM_TOKENS_DIR", music_lm.get("maestro_tokens_dir")),
        shell_default("MUSIC_LM_OUTPUT_DIR", music_lm.get("output_dir")),
        shell_default("MUSIC_LM_CHECKPOINT", music_lm.get("checkpoint")),
        shell_default("MUSIC_LM_VENV", music_lm.get("venv")),
        shell_default("MUSIC_LM_PYTHON_BIN", music_lm.get("python_bin")),
        shell_default("MUSIC_LM_TIME_STEP_SECONDS", music_lm.get("time_step_seconds")),
        shell_default("MUSIC_LM_MAX_TIME_SHIFT_STEPS", music_lm.get("max_time_shift_steps")),
        shell_default("MUSIC_LM_VELOCITY_BINS", music_lm.get("velocity_bins")),
        shell_default("MUSIC_LM_BLOCK_SIZE", music_lm.get("block_size")),
        shell_default("MUSIC_LM_BATCH_SIZE", music_lm.get("batch_size")),
        shell_default("MUSIC_LM_N_LAYER", music_lm.get("n_layer")),
        shell_default("MUSIC_LM_N_HEAD", music_lm.get("n_head")),
        shell_default("MUSIC_LM_N_EMBD", music_lm.get("n_embd")),
        shell_default("MUSIC_LM_DROPOUT", music_lm.get("dropout")),
        shell_default("MUSIC_LM_LEARNING_RATE", music_lm.get("learning_rate")),
        shell_default("MUSIC_LM_WEIGHT_DECAY", music_lm.get("weight_decay")),
        shell_default("MUSIC_LM_MAX_STEPS", music_lm.get("max_steps")),
        shell_default("MUSIC_LM_EVAL_INTERVAL", music_lm.get("eval_interval")),
        shell_default("MUSIC_LM_EVAL_ITERS", music_lm.get("eval_iters")),
        shell_default("MUSIC_LM_GRAD_CLIP", music_lm.get("grad_clip")),
        shell_default("MUSIC_LM_SEED", music_lm.get("seed")),
        shell_default("MUSIC_LM_PPO_REWARD_WEIGHT", music_lm.get("ppo_reward_weight")),
        shell_default("MUSIC_LM_PPO_REWARD_WINDOW_TOKENS", music_lm.get("ppo_reward_window_tokens")),
        shell_default("MUSIC_LM_PPO_REWARD_CLIP", music_lm.get("ppo_reward_clip")),
    ]


EMITTERS = {
    "paths": emit_paths,
    "environment": emit_environment,
    "scheduler": emit_scheduler,
    "artifacts": emit_artifacts,
    "music_lm": emit_music_lm,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("sections", nargs="*", default=["paths"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    lines: list[str] = []
    for name in args.sections:
        if name not in EMITTERS:
            raise SystemExit(f"Unknown config export section: {name}")
        lines.extend(EMITTERS[name](cfg))
    print("\n".join(line for line in lines if line))


if __name__ == "__main__":
    main()
