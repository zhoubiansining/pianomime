#!/usr/bin/env python3
"""
Plot training curves from eval_history.jsonl files.

Usage:
    # Plot a single run
    python experiments/plot_training_curves.py --exp-dir /path/to/baseline__TwinkleTwinkleRousseau

    # Compare multiple runs (one run each)
    python experiments/plot_training_curves.py \
        --runs baseline__TwinkleTwinkleRousseau \
               smooth_combo__TwinkleTwinkleRousseau \
        --root-dir /tmp/pianomime_reward_sweep \
        --output comparison.png

    # Plot F1, precision, recall on separate subplots
    python experiments/plot_training_curves.py --exp-dir /path/to/run --metrics f1 precision recall
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METRICS = ["f1", "precision", "recall", "eval_reward"]


def load_eval_history(exp_dir: Path):
    history_path = exp_dir / "eval_history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"eval_history.jsonl not found in {exp_dir}")

    rows = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_curves(
    runs: list[dict],
    metrics: list[str],
    output: Path,
    title: str = "",
    smooth_window: int = 1,
):
    """Plot training curves for multiple runs.

    Args:
        runs: List of dicts with keys 'label', 'data' (list of eval dicts).
        metrics: Which metrics to plot.
        output: Output PNG path.
        title: Figure title.
        smooth_window: Moving-average window for smoothing.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(10, 4 * n_metrics),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx, 0]
        for run_idx, run in enumerate(runs):
            data = run["data"]
            if not data:
                continue
            x = [d.get("iter", i) for i, d in enumerate(data)]
            y_raw = [d.get(metric) for d in data]

            # Filter None values
            x_clean = []
            y_clean = []
            for xi, yi in zip(x, y_raw):
                if yi is not None:
                    x_clean.append(xi)
                    y_clean.append(yi)

            if not x_clean:
                continue

            y_arr = np.array(y_clean, dtype=float)
            x_arr = np.array(x_clean)

            if smooth_window > 1:
                kernel = np.ones(smooth_window) / smooth_window
                y_smooth = np.convolve(y_arr, kernel, mode="same")
                # Draw raw as faint background, smoothed as main line
                ax.plot(x_arr, y_arr, color=colors[run_idx], alpha=0.15, linewidth=0.8)
                ax.plot(x_arr, y_smooth, color=colors[run_idx],
                        linewidth=2, label=run["label"])
            else:
                ax.plot(x_arr, y_arr, color=colors[run_idx],
                        linewidth=1.5, label=run["label"])

            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {output}")


def plot_single(exp_dir: Path, output: Path, metrics: list[str], smooth_window: int = 5):
    run_name = exp_dir.name
    data = load_eval_history(exp_dir)
    if not data:
        print(f"[WARN] No data in {exp_dir}")
        return
    runs = [{"label": run_name, "data": data}]
    title = f"Training Curves: {run_name}"
    plot_curves(runs, metrics, output, title, smooth_window)


def plot_comparison(
    root_dir: Path,
    run_dirs: list[str],
    output: Path,
    metrics: list[str],
    smooth_window: int = 5,
):
    runs = []
    for rd in run_dirs:
        exp_dir = root_dir / rd
        if not exp_dir.exists():
            print(f"[WARN] Run directory not found: {exp_dir}")
            continue
        try:
            data = load_eval_history(exp_dir)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        runs.append({"label": rd, "data": data})

    if not runs:
        print("[ERROR] No valid runs found.")
        sys.exit(1)

    title = f"Reward Shaping Comparison: {', '.join(r['label'] for r in runs)}"
    plot_curves(runs, metrics, output, title, smooth_window)


def main():
    parser = argparse.ArgumentParser(description="Plot PianoMime training curves")
    parser.add_argument("--exp-dir", type=Path,
                        help="Single experiment directory (contains eval_history.jsonl)")
    parser.add_argument("--runs", nargs="*",
                        help="Run directory names for comparison (use with --root-dir)")
    parser.add_argument("--root-dir", type=Path, default=Path("/tmp/pianomime_reward_sweep"),
                        help="Root directory containing run subdirs")
    parser.add_argument("--output", type=Path,
                        help="Output PNG path (default: stdout plot to --exp-dir/curve.png)")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS,
                        help=f"Metrics to plot (default: {' '.join(DEFAULT_METRICS)})")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Smoothing window size (default: 5)")
    args = parser.parse_args()

    if args.exp_dir:
        output = args.output or (args.exp_dir / "curve.png")
        plot_single(args.exp_dir, output, args.metrics, args.smooth)
    elif args.runs:
        output = args.output or (args.root_dir / "comparison.png")
        plot_comparison(args.root_dir, args.runs, output, args.metrics, args.smooth)
    else:
        print("Specify either --exp-dir or --runs")
        sys.exit(1)


if __name__ == "__main__":
    main()
