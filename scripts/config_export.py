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
        shell_default("PPO_BLOCKED_TASKS", single.get("ppo_blocked_songs")),
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


EMITTERS = {
    "paths": emit_paths,
    "environment": emit_environment,
    "scheduler": emit_scheduler,
    "artifacts": emit_artifacts,
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
