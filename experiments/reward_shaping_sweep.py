#!/usr/bin/env python3
"""
Reward Shaping Sweep for PianoMime single-song policy.

Trains multiple reward variants across multiple songs, then produces
a comparison report with aggregate and per-run metrics.

Usage:
    python reward_shaping_sweep.py --songs TwinkleTwinkleRousseau Pirates_1 --variants baseline vel_smooth
    CUDA_VISIBLE_DEVICES=0,1,2,3 python reward_shaping_sweep.py  # 4-way parallel on 4 GPUs
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robopianist import music

# The sweep chdirs into pianomime/ before running train_ppo.py,
# so relative paths like "dataset/..." work from there.
# Override with --dataset-root if your data lives elsewhere.
DATASET_ROOT_DEFAULT = "dataset"

# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_SONGS = [
    "TwinkleTwinkleRousseau",
    "Pirates_1",
    "Stan_1",
    "Petrunko_3",
]

# All reward shaping variants to compare.
# Each entry: variant_name -> { arg_name: value, ... }
# Boolean True -> add flag only; Boolean False -> omit flag.
# String/Number -> add flag with value.
DEFAULT_VARIANTS: Dict[str, Dict[str, Any]] = {
    # --- baselines ---
    "baseline": {},

    # --- single reward components ---
    "vel_smooth": {
        "reward_velocity_smoothness": True,
        "reward_velocity_smoothness_coef": 0.01,
    },
    "action_smooth": {
        "reward_action_smoothness": True,
        "reward_action_smoothness_coef": 0.01,
    },
    "preposition": {
        "reward_prepositioning": True,
        "reward_prepositioning_coef": 0.3,
        "reward_prepositioning_lookahead": 5,
    },
    "timing": {
        "reward_timing": True,
        "reward_timing_coef": 0.5,
    },
    "finger_dist": {
        "reward_finger_key_distance": True,
        "reward_finger_key_distance_coef": 0.5,
    },
    "collision": {
        "reward_finger_collision": True,
        "reward_finger_collision_coef": 0.5,
    },

    # --- combinations ---
    "smooth_combo": {
        "reward_velocity_smoothness": True,
        "reward_velocity_smoothness_coef": 0.01,
        "reward_action_smoothness": True,
        "reward_action_smoothness_coef": 0.01,
    },
    "full_combo": {
        "reward_velocity_smoothness": True,
        "reward_velocity_smoothness_coef": 0.01,
        "reward_action_smoothness": True,
        "reward_action_smoothness_coef": 0.01,
        "reward_prepositioning": True,
        "reward_prepositioning_coef": 0.3,
        "reward_prepositioning_lookahead": 5,
        "reward_finger_key_distance": True,
        "reward_finger_key_distance_coef": 0.5,
        "reward_finger_collision": True,
        "reward_finger_collision_coef": 0.5,
    },
}


# =============================================================================
# Helpers
# =============================================================================

def to_cli_flag(key: str) -> str:
    """Convert Python arg name to CLI flag (e.g. foo_bar -> --foo-bar)."""
    return "--" + key.replace("_", "-")


def variant_dir_name(variant: str, song: str, seed: int, multi_seed: bool) -> str:
    if multi_seed:
        return f"{variant}__{song}__seed{seed}"
    return f"{variant}__{song}"


def build_command(
    train_script: str,
    base_args: List[str],
    variant_overrides: Dict[str, Any],
    seed: int,
    song: str,
    run_name: str,
) -> Tuple[List[str], Dict[str, str]]:
    """Build the full train_ppo.py command line and its env vars."""
    cmd = [
        sys.executable,
        train_script,
        *base_args,
        "--seed", str(seed),
        "--mimic-task", song,
        "--environment-name", song,
        "--name", run_name,
    ]
    for key, value in variant_overrides.items():
        flag = to_cli_flag(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)  # add flag only for True
            # False -> omit flag (use default)
        elif isinstance(value, list):
            for v in value:
                cmd.extend([flag, str(v)])
        else:
            cmd.extend([flag, str(value)])

    # Environment variables
    env = os.environ.copy()
    # MuJoCo rendering
    env.setdefault("MUJOCO_GL", "egl")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Suppress TF / JAX info spam
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return cmd, env


def uses_note_trajectory(song: str, dataset_root: Path) -> bool:
    """Decide whether a song should be loaded via note trajectory.

    For built-in RoboPianist songs, `music.load(song)` works.
    For dataset-specific names (e.g. `Pirates_1`), use `--use-note-trajectory`.
    """
    try:
        music.load(song)
        return False
    except Exception:
        note_path = dataset_root / "notes" / f"{song}.pkl"
        return note_path.exists()


def run_training(
    cmd: List[str],
    env: Dict[str, str],
    run_name: str,
    gpu_id: Optional[int],
    dry_run: bool,
) -> int:
    """Execute a single training run. Returns exit code."""
    if gpu_id is not None:
        gpu_env = env.copy()
        gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_env["MUJOCO_EGL_DEVICE_ID"] = "0"
    else:
        gpu_env = env

    print(f"[RUN] {run_name}" + (f" GPU={gpu_id}" if gpu_id is not None else ""))
    print(f"      {' '.join(cmd)}")
    if dry_run:
        return 0

    result = subprocess.run(cmd, env=gpu_env)
    if result.returncode != 0:
        print(f"[FAIL] {run_name} (exit {result.returncode})", file=sys.stderr)
    else:
        print(f"[DONE] {run_name}")
    return result.returncode


def collect_results(root_dir: Path, variants: List[str], songs: List[str], seeds: List[int]) -> List[Dict]:
    """Read result.json files produced by train_ppo.py."""
    multi_seed = len(seeds) > 1
    rows = []
    for variant, song, seed in itertools.product(variants, songs, seeds):
        run_dir = root_dir / variant_dir_name(variant, song, seed, multi_seed)
        result_path = run_dir / "result.json"
        if not result_path.exists():
            rows.append({
                "variant": variant,
                "song": song,
                "seed": seed,
                "status": "missing",
            })
            continue
        try:
            data = json.loads(result_path.read_text())
        except Exception as e:
            rows.append({
                "variant": variant,
                "song": song,
                "seed": seed,
                "status": f"parse_error: {e}",
            })
            continue

        metrics = data.get("metrics", {})
        statistics = data.get("statistics", {})
        rows.append({
            "variant": variant,
            "song": song,
            "seed": seed,
            "status": "ok",
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "sustain_f1": metrics.get("sustain_f1"),
            "total_reward": data.get("total_reward"),
            "episode_length": statistics.get("episode_length") if statistics else None,
            "best_video": data.get("best_video"),
            "final_video": data.get("final_video"),
            "eval_history": data.get("eval_history"),
            "run_dir": str(run_dir),
        })
    return rows


def has_valid_result(run_dir: Path) -> bool:
    """Return True if an existing run already has a parseable result.json."""
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return False
    try:
        data = json.loads(result_path.read_text())
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("metrics"), dict)


def summarize(rows: List[Dict]) -> List[Dict]:
    """Aggregate metrics by variant."""
    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(row["variant"], []).append(row)

    summary = []
    for variant, items in grouped.items():
        def avg(key: str):
            vals = [item[key] for item in items if item.get(key) is not None]
            return mean(vals) if vals else None

        summary.append({
            "variant": variant,
            "num_runs": len(items),
            "avg_f1": avg("f1"),
            "avg_precision": avg("precision"),
            "avg_recall": avg("recall"),
            "avg_sustain_f1": avg("sustain_f1"),
            "avg_total_reward": avg("total_reward"),
            "avg_episode_length": avg("episode_length"),
        })

    # Sort by F1 descending
    summary.sort(key=lambda x: (
        -1 if x["avg_f1"] is None else -x["avg_f1"],
        x["variant"],
    ))
    return summary


def write_report(root_dir: Path, rows: List[Dict], summary: List[Dict], config: Dict) -> Path:
    """Write markdown summary report."""
    report_path = root_dir / "summary.md"
    lines = []
    lines.append("# Reward Shaping Sweep Summary\n")

    # Config
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Songs:** `{', '.join(config['songs'])}`")
    lines.append(f"- **Variants:** `{', '.join(config['variants'])}`")
    lines.append(f"- **Seeds:** `{', '.join(map(str, config['seeds']))}`")
    lines.append(f"- **Total iters:** `{config['total_iters']}`")
    lines.append(f"- **Num envs:** `{config['num_envs']}`")
    lines.append(f"- **Residual factor:** `{config['residual_factor']}`")
    lines.append("")

    # Aggregate table
    lines.append("## Aggregate Results (sorted by avg F1)")
    lines.append("")
    header = "| variant | runs | avg_f1 | avg_precision | avg_recall | avg_reward | avg_ep_len |"
    sep    = "|---|---|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for item in summary:
        f1 = f"{item['avg_f1']:.4f}" if item['avg_f1'] is not None else "n/a"
        pr = f"{item['avg_precision']:.4f}" if item['avg_precision'] is not None else "n/a"
        rc = f"{item['avg_recall']:.4f}" if item['avg_recall'] is not None else "n/a"
        rw = f"{item['avg_total_reward']:.2f}" if item['avg_total_reward'] is not None else "n/a"
        ep = f"{item['avg_episode_length']:.1f}" if item['avg_episode_length'] is not None else "n/a"
        lines.append(f"| {item['variant']} | {item['num_runs']} | {f1} | {pr} | {rc} | {rw} | {ep} |")
    lines.append("")

    # Per-run table
    lines.append("## Per-Run Results")
    lines.append("")
    lines.append("| variant | song | seed | status | f1 | precision | recall | reward | best_video |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row['variant']} | {row['song']} | {row['seed']} | "
                f"{row['status']} | n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            f1_str = f"{row['f1']:.4f}" if row.get("f1") is not None else "n/a"
            pr_str = f"{row['precision']:.4f}" if row.get("precision") is not None else "n/a"
            rc_str = f"{row['recall']:.4f}" if row.get("recall") is not None else "n/a"
            rw_str = f"{row['total_reward']:.2f}" if row.get("total_reward") is not None else "n/a"
            video = row.get("best_video")
            video_str = f"[video]({Path(video).name})" if video else "n/a"
            lines.append(
                f"| {row['variant']} | {row['song']} | {row['seed']} | ok | "
                f"{f1_str} | {pr_str} | {rc_str} | {rw_str} | {video_str} |"
            )
    lines.append("")
    report_path.write_text("\n".join(lines))
    return report_path


def _load_eval_history(run_dir: Path) -> List[Dict]:
    """Load per-iteration eval metrics for plotting curves."""
    path = run_dir / "eval_history.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def plot_sweep_curves(
    root_dir: Path,
    rows: List[Dict],
    variants: List[str],
    songs: List[str],
    seeds: List[int],
) -> None:
    """Plot training curves comparing variants, one subplot per song."""
    n_songs = len(songs)
    fig, axes = plt.subplots(
        n_songs, 1,
        figsize=(11, 4.5 * n_songs),
        squeeze=False,
    )
    fig.suptitle(
        "F1 Training Curves: Reward Shaping Variants",
        fontsize=14, fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 10)))

    for song_idx, song in enumerate(songs):
        ax = axes[song_idx, 0]
        ax.set_title(f"Song: {song}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Eval F1")
        ax.grid(True, alpha=0.3)

        for variant_idx, variant in enumerate(variants):
            # Average over seeds for this variant×song
            seed_curves: List[List[float]] = []
            for seed in seeds:
                run_name = variant_dir_name(variant, song, seed, len(seeds) > 1)
                hist = _load_eval_history(root_dir / run_name)
                if hist:
                    f1s = [h.get("f1", 0.0) for h in hist if h.get("f1") is not None]
                    if f1s:
                        seed_curves.append(f1s)

            if not seed_curves:
                continue

            # Trim all curves to the shortest length so we can average
            min_len = min(len(c) for c in seed_curves)
            arr = np.array([c[:min_len] for c in seed_curves], dtype=float)
            mean_curve = arr.mean(axis=0)
            x = np.arange(min_len)

            ax.plot(x, mean_curve, color=colors[variant_idx % len(colors)],
                    linewidth=2, label=variant)
            if arr.shape[0] > 1:
                std = arr.std(axis=0)
                ax.fill_between(x, mean_curve - std, mean_curve + std,
                                color=colors[variant_idx % len(colors)], alpha=0.15)

        ax.legend(loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    out_path = root_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[DONE] Curves: {out_path}")


# =============================================================================
# Parallel execution
# =============================================================================

def run_parallel(
    jobs: List[Tuple[str, List[str], Dict[str, str]]],
    gpu_ids: List[int],
    dry_run: bool,
) -> List[Tuple[str, int]]:
    """Launch training jobs in parallel, keeping at most len(gpu_ids) running."""
    failures: List[Tuple[str, int]] = []
    if not jobs:
        return failures

    # Each running entry: (run_name, popen, gpu_id)
    active: List[Tuple[str, subprocess.Popen, int]] = []
    job_idx = 0
    spawned = 0

    while job_idx < len(jobs) or active:
        # Launch until all GPU slots filled
        while len(active) < len(gpu_ids) and job_idx < len(jobs):
            run_name, cmd, env = jobs[job_idx]
            gpu_id = gpu_ids[spawned % len(gpu_ids)]
            spawned += 1
            job_idx += 1

            env_with_gpu = env.copy()
            env_with_gpu["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env_with_gpu["MUJOCO_EGL_DEVICE_ID"] = "0"

            print(f"[SPAWN] [{job_idx}/{len(jobs)}] {run_name} GPU={gpu_id}")
            if dry_run:
                print(f"        {' '.join(cmd)}")
                # Use a sentinel for dry-run accounting
                active.append((run_name, None, gpu_id))  # type: ignore
                continue
            proc = subprocess.Popen(cmd, env=env_with_gpu)
            active.append((run_name, proc, gpu_id))

        if not active:
            break

        # Wait for the oldest job to finish (FIFO)
        run_name, proc, gpu_id = active.pop(0)
        if proc is None:  # dry run
            continue
        code = proc.wait()
        if code != 0:
            print(f"[FAIL] {run_name} (exit {code}, GPU={gpu_id})", file=sys.stderr)
            failures.append((run_name, code))
        else:
            print(f"[DONE] {run_name} (GPU={gpu_id})")

    return failures


def run_sequential(
    jobs: List[Tuple[str, List[str], Dict[str, str]]],
    gpu_ids: List[int],
    dry_run: bool,
) -> List[Tuple[str, int]]:
    """Run jobs one at a time."""
    failures: List[Tuple[str, int]] = []
    for idx, (run_name, cmd, env) in enumerate(jobs):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        code = run_training(cmd, env, run_name, gpu_id, dry_run)
        if code != 0:
            failures.append((run_name, code))
    return failures


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PianoMime reward shaping sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--songs", nargs="*", default=DEFAULT_SONGS,
                        help="Songs to evaluate")
    parser.add_argument("--variants", nargs="*",
                        default=list(DEFAULT_VARIANTS.keys()),
                        help="Reward variants to compare")
    parser.add_argument("--reward-config", type=Path, default=None,
                        help="JSON file defining/overriding reward variant coefficients "
                             "(merged on top of built-in defaults)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42],
                        help="Random seeds (each seed runs every song×variant)")
    parser.add_argument("--root-dir", default="/tmp/pianomime_reward_sweep",
                        help="Root output directory")
    parser.add_argument("--train-script",
                        default="single_task/train_ppo.py",
                        help="Path to train_ppo.py (relative to current working directory; CWD should be pianomime/)")
    parser.add_argument("--total-iters", type=int, default=300,
                        help="Training iterations per run")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N iterations (default: 10)")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel envs")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="PPO batch_size")
    parser.add_argument("--residual-factor", type=float, default=0.03,
                        help="Residual factor")
    parser.add_argument("--cuda-devices", nargs="*", type=int, default=[0],
                        help="GPU device IDs for parallel execution")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially (one at a time)")
    parser.add_argument("--base-arg", action="append", default=[],
                        help="Extra args passed to every train_ppo.py invocation")
    parser.add_argument("--rerun", action="store_true",
                        help="Force rerun even if result.json already exists")

    args = parser.parse_args()

    # Load reward config and merge with built-in defaults
    variants_registry = dict(DEFAULT_VARIANTS)  # shallow copy
    if args.reward_config is not None:
        config_path = Path(args.reward_config)
        if not config_path.exists():
            print(f"[ERROR] Reward config not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(config_path) as f:
            config_overrides = json.load(f)
        for name, overrides in config_overrides.items():
            if name in variants_registry:
                # Merge: override existing variant coefficients
                variants_registry[name] = {**variants_registry[name], **overrides}
            else:
                # New variant
                variants_registry[name] = overrides
        print(f"[INFO] Loaded reward config from {config_path}")

    # Validate variants
    for v in args.variants:
        if v not in variants_registry:
            print(f"[ERROR] Unknown variant: {v}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(variants_registry))}", file=sys.stderr)
            sys.exit(1)

    # Validate songs (relative to current working directory)
    dataset_root = Path(DATASET_ROOT_DEFAULT)
    missing = []
    missing_note = []
    for song in args.songs:
        lh = dataset_root / "high_level_trajectories" / f"{song}_left_hand_action_list.npy"
        rh = dataset_root / "high_level_trajectories" / f"{song}_right_hand_action_list.npy"
        note = dataset_root / "notes" / f"{song}.pkl"
        if not lh.exists() or not rh.exists():
            missing.append(song)
        if not note.exists():
            missing_note.append(song)
    if missing:
        print(f"[WARN] Dataset files not found for: {', '.join(missing)}")
        print(f"[WARN] Searched in: {dataset_root.absolute()}")
        print(f"[WARN] Make sure to run from the pianomime/ directory.")
    if missing_note:
        print(f"[WARN] Note trajectories not found for: {', '.join(missing_note)}")

    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Base args for train_ppo.py
    base_args = [
        "--root-dir", str(root_dir),
        "--max-steps", "10000000",
        "--discount", "0.99",
        "--n-steps-lookahead", "10",
        "--tqdm-bar",
        "--eval-episodes", "1",
        "--camera-id", "piano/back",
        "--midi-start-from", "0",
        "--num-envs", str(args.num_envs),
        "--initial-lr", "3e-4",
        "--lr-decay-rate", "0.999",
        "--n-steps", str(args.n_steps),
        "--batch-size", str(args.batch_size),
        "--total-iters", str(args.total_iters),
        "--eval-interval", str(args.eval_interval),
        "--residual-factor", str(args.residual_factor),
        "--residual-action",
        "--gravity-compensation",
    ] + args.base_arg

    # Build job list
    multi_seed = len(args.seeds) > 1
    jobs = []
    skipped = 0
    for seed, song, variant in itertools.product(args.seeds, args.songs, args.variants):
        overrides = variants_registry[variant]
        run_name = variant_dir_name(variant, song, seed, multi_seed)
        run_dir = root_dir / run_name

        if (not args.rerun) and has_valid_result(run_dir):
            skipped += 1
            print(f"[SKIP] {run_name} (existing result.json)")
            continue

        cmd, env = build_command(
            train_script=args.train_script,
            base_args=base_args,
            variant_overrides=overrides,
            seed=seed,
            song=song,
            run_name=run_name,
        )
        if uses_note_trajectory(song, dataset_root):
            cmd.append("--use-note-trajectory")
        jobs.append((run_name, cmd, env))

    total_jobs = len(jobs)
    print(f"\n[INFO] {total_jobs} runs to execute")
    if skipped:
        print(f"[INFO] {skipped} runs skipped (existing results)")
    print(f"[INFO] Parallelism: {len(args.cuda_devices)} GPU(s)")
    print(f"[INFO] Mode: {'sequential' if args.sequential else 'parallel'}\n")

    # Execute
    if args.sequential or len(args.cuda_devices) == 1:
        failures = run_sequential(jobs, args.cuda_devices, args.dry_run)
    else:
        failures = run_parallel(jobs, args.cuda_devices, args.dry_run)

    if args.dry_run:
        print("\n[DONE] Dry run complete.")
        return

    # Collect and report
    print("\n[INFO] Collecting results...")
    rows = collect_results(root_dir, args.variants, args.songs, args.seeds)
    summary = summarize(rows)

    config = {
        "songs": args.songs,
        "variants": args.variants,
        "seeds": args.seeds,
        "total_iters": args.total_iters,
        "num_envs": args.num_envs,
        "residual_factor": args.residual_factor,
    }

    # Write JSON
    summary_json_path = root_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps({"rows": rows, "summary": summary, "config": config}, indent=2)
    )

    # Write Markdown
    report_path = write_report(root_dir, rows, summary, config)

    print(f"[DONE] Report: {report_path}")
    print(f"[DONE] JSON:   {summary_json_path}")
    if failures:
        print(f"[WARN] {len(failures)} run(s) failed during training; see summary for missing results.")

    # Generate comparison training curves plot
    try:
        plot_sweep_curves(root_dir, rows, args.variants, args.songs, args.seeds)
    except Exception as e:
        print(f"[WARN] Failed to generate curves plot: {e}")

    print("\n=== Aggregate Results ===")
    print(f"{'variant':<20} {'avg_f1':>10} {'runs':>6}")
    print("-" * 38)
    for item in summary:
        f1 = f"{item['avg_f1']:.4f}" if item['avg_f1'] is not None else "n/a"
        print(f"{item['variant']:<20} {f1:>10} {item['num_runs']:>6}")


if __name__ == "__main__":
    main()
