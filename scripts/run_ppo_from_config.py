#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pianomime_config import cli_args_from_mapping, load_config, section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("song", nargs="?")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--music-lm-checkpoint", default=None)
    parser.add_argument("--music-lm-reward-weight", type=float, default=None)
    parser.add_argument("--music-lm-reward-window-tokens", type=int, default=None)
    parser.add_argument("--music-lm-reward-clip", type=float, default=None)
    parser.add_argument("--music-lm-reference-log-ppl", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = section(cfg, "paths")
    ppo = dict(section(cfg, "single_song", "ppo"))
    songs = section(cfg, "single_song").get("ppo_songs", [])
    if args.song:
        song = args.song
    elif songs:
        song = songs[0]
    else:
        raise SystemExit("No PPO song was provided and single_song.ppo_songs is empty.")
    run_id = os.environ.get("RUN_ID", "manual")
    run_name = args.run_name or f"{song}_ppo_curve_{run_id}"
    root_dir = args.root_dir or ppo.pop("root_dir", None) or f"{paths['local_results_dir']}/single_song/training_runs"

    command = [
        os.environ.get("PYTHON_BIN", paths.get("python_bin", sys.executable)),
        str(PROJECT_ROOT / "single_task" / "train_ppo.py"),
        "--root-dir",
        str(root_dir),
        "--name",
        run_name,
        "--mimic-task",
        song,
        "--environment-name",
        song,
    ]
    if args.pretrained:
        command.extend(["--pretrained", args.pretrained])
    if args.music_lm_checkpoint:
        command.extend(["--music-lm-checkpoint", args.music_lm_checkpoint])
    if args.music_lm_reward_weight is not None:
        command.extend(["--music-lm-reward-weight", str(args.music_lm_reward_weight)])
    if args.music_lm_reward_window_tokens is not None:
        command.extend(["--music-lm-reward-window-tokens", str(args.music_lm_reward_window_tokens)])
    if args.music_lm_reward_clip is not None:
        command.extend(["--music-lm-reward-clip", str(args.music_lm_reward_clip)])
    if args.music_lm_reference_log_ppl is not None:
        command.extend(["--music-lm-reference-log-ppl", str(args.music_lm_reference_log_ppl)])
    for handled_key in ("root_dir", "name", "mimic_task", "environment_name", "pretrained"):
        ppo.pop(handled_key, None)
    command.extend(cli_args_from_mapping(ppo))

    if args.dry_run:
        print(" ".join(command))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'single_task'}:{env.get('PYTHONPATH', '')}"
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
