#!/usr/bin/env python3
"""
Analyze and visualize sensitivity results from run_sensitivity.sh.

Reads all eval_history.jsonl files from the sensitivity results directory
and produces publication-quality figures.

Usage:
    python plot_sensitivity.py [--results-dir DIR] [--output-dir DIR]
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/Users/Sining/Desktop/Files/Course/Homework/2026春季/深度强化学习/Project/pianomime/results/reward_sensitivity")
OUT  = ROOT

# ── colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "timing":        "#1f77b4",   # blue
    "preposition":   "#ff7f0e",   # orange
    "vel_smooth":    "#2ca02c",   # green
    "act_smooth":    "#d62728",   # red
    "finger_dist":   "#9467bd",   # purple
}
COEF_LINESTYLE = {              # coefficient → line style / marker
    0.001:  (":",   "o"),
    0.01:   ("-",   "s"),
    0.1:    ("--",  "^"),
    0.3:    ("-",   "D"),
    0.5:    ("-.",  "v"),
    1.0:    (":",   "p"),
    2.0:    ("--",  "*"),
}

# ── helpers ───────────────────────────────────────────────────────────────────
def load_history(run_dir: Path) -> list[dict]:
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


def smooth(y, sigma=2):
    return gaussian_filter1d(y, sigma=sigma)


def group_variants(variants: list[str]) -> dict[str, list[str]]:
    """Map group label → list of variant names in that group."""
    groups = {}
    for v in variants:
        if v == "baseline":
            continue
        if v.startswith("timing"):
            groups.setdefault("timing", []).append(v)
        elif v.startswith("prep_"):
            groups.setdefault("preposition", []).append(v)
        elif v.startswith("vel_sm_"):
            groups.setdefault("vel_smooth", []).append(v)
        elif v.startswith("act_sm_"):
            groups.setdefault("act_smooth", []).append(v)
        elif v.startswith("fdist_"):
            groups.setdefault("finger_dist", []).append(v)
    return groups


def parse_coef(variant: str) -> float:
    for suffix in ("_0.001","_0.01","_0.1","_0.3","_0.5","_1.0","_2.0"):
        if variant.endswith(suffix):
            return float("0" + suffix[1:])
    return None


# ── figure helpers ─────────────────────────────────────────────────────────────
def setup_ax(ax, title, ylabel, xlabel="Training Iteration"):
    ax.set_title(title, pad=10, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=9)


# ── 1. Training curves (one subplot per reward group) ─────────────────────────
def plot_training_curves(results_dir: Path, out_dir: Path,
                         eval_interval: int = 10, smooth_sigma: int = 2):
    all_dirs = list(results_dir.glob("*__*"))
    variants = sorted(set(d.name.split("__")[0] for d in all_dirs))
    groups = group_variants(variants)

    baseline_dir = results_dir / "baseline__TwinkleTwinkleRousseau"
    baseline_hist = load_history(baseline_dir)
    if baseline_hist:
        b_iters = np.array([h.get("iter", i) for i, h in enumerate(baseline_hist)])
        b_f1    = np.array([h.get("f1", 0) for h in baseline_hist])
        b_smooth = smooth(b_f1, smooth_sigma)
    else:
        b_iters = np.array([])
        b_smooth = np.array([])

    n_groups = len(groups)
    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(10, 3.8 * n_groups),
        squeeze=False,
    )
    fig.suptitle(
        "Reward Shaping Sensitivity: F1 vs. Training Iteration",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(hspace=0.42)

    for row, (group, member_vars) in enumerate(sorted(groups.items())):
        ax = axes[row, 0]
        color = PALETTE.get(group, "#333333")

        # Sort members by coefficient
        member_vars_sorted = sorted(member_vars, key=parse_coef)
        for mv in member_vars_sorted:
            coef = parse_coef(mv)
            ls, mk = COEF_LINESTYLE.get(coef, ("-", "o"))

            run_dir = results_dir / f"{mv}__TwinkleTwinkleRousseau"
            hist = load_history(run_dir)
            if not hist:
                continue
            iters = np.array([h.get("iter", i) for i, h in enumerate(hist)])
            f1    = np.array([h.get("f1", 0) for h in hist])
            f1_s  = smooth(f1, smooth_sigma)

            label = f"coef={coef}"
            ax.plot(iters, f1_s, color=color, linestyle=ls,
                    linewidth=1.8, label=label, alpha=0.9)

        # Baseline
        if b_iters.size:
            ax.plot(b_iters, b_smooth, color="black", linestyle="-",
                    linewidth=2.2, label="baseline", alpha=0.85, zorder=10)

        setup_ax(ax, f"Reward: {group}", "Eval F1 (smoothed)")
        ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.9)

    plt.savefig(out_dir / "sensitivity_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] sensitivity_training_curves.png")


# ── 2. Bar chart: best F1 per group / coefficient ─────────────────────────────
def plot_best_f1_bars(results_dir: Path, out_dir: Path):
    all_dirs = list(results_dir.glob("*__*"))
    variants = sorted(set(d.name.split("__")[0] for d in all_dirs))
    groups = group_variants(variants)

    baseline_dir = results_dir / "baseline__TwinkleTwinkleRousseau"
    b_hist = load_history(baseline_dir)
    baseline_f1 = max((h.get("f1", 0) for h in b_hist), default=0)

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(3.6 * n_groups, 4), squeeze=False)

    for col, (group, member_vars) in enumerate(sorted(groups.items())):
        ax = axes[0, col]
        color = PALETTE.get(group, "#333333")

        member_vars_sorted = sorted(member_vars, key=parse_coef)
        coefs   = [parse_coef(mv) for mv in member_vars_sorted]
        best_f1s = []
        for mv in member_vars_sorted:
            run_dir = results_dir / f"{mv}__TwinkleTwinkleRousseau"
            hist = load_history(run_dir)
            best = max((h.get("f1", 0) for h in hist), default=0)
            best_f1s.append(best)

        x = np.arange(len(coefs))
        bars = ax.bar(x, best_f1s, color=color, alpha=0.8, edgecolor="white", linewidth=0.6)

        # Baseline horizontal line
        ax.axhline(baseline_f1, color="black", linestyle="--",
                   linewidth=1.5, label=f"baseline ({baseline_f1:.3f})", zorder=10)

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in coefs], fontsize=9)
        ax.set_xlabel("Coefficient", fontsize=10)
        ax.set_ylabel("Best Eval F1", fontsize=10)
        ax.set_title(group.replace("_", " ").title(), pad=8, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="lower right")

        # Value labels on bars
        for bar, val in zip(bars, best_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_ylim(0, max(max(best_f1s), baseline_f1) * 1.12)

    fig.suptitle("Reward Shaping Sensitivity: Best F1 by Coefficient",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_best_f1_bars.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] sensitivity_best_f1_bars.png")


# ── 3. Heatmap: F1 matrix (groups × coefficients) ─────────────────────────────
def plot_f1_heatmap(results_dir: Path, out_dir: Path):
    all_dirs = list(results_dir.glob("*__*"))
    variants = sorted(set(d.name.split("__")[0] for d in all_dirs))
    groups = group_variants(variants)

    baseline_dir = results_dir / "baseline__TwinkleTwinkleRousseau"
    b_hist = load_history(baseline_dir)
    baseline_f1 = max((h.get("f1", 0) for h in b_hist), default=0)

    # Collect all unique coefficients across all groups first (union)
    all_coefs: set[float] = set()
    for group, mvs in groups.items():
        for mv in mvs:
            c = parse_coef(mv)
            if c is not None:
                all_coefs.add(c)
    col_labels = sorted(all_coefs)
    col_map = {c: i for i, c in enumerate(col_labels)}

    n_rows = len(groups)
    n_cols = len(col_labels)
    mat = np.full((n_rows, n_cols), np.nan)
    row_labels = []

    for row_idx, (group, mvs) in enumerate(sorted(groups.items())):
        row_labels.append(group.replace("_", " ").title())
        for mv in mvs:
            coef = parse_coef(mv)
            if coef is None:
                continue
            col_idx = col_map[coef]
            run_dir = results_dir / f"{mv}__TwinkleTwinkleRousseau"
            hist = load_history(run_dir)
            best = max((h.get("f1", 0) for h in hist), default=0)
            mat[row_idx, col_idx] = best

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * n_cols + 2), 0.55 * n_rows + 2))

    # Δ F1 from baseline
    delta = mat - baseline_f1
    if np.all(np.isnan(delta)):
        vmax = 0.1
    else:
        vmax = max(abs(delta.min()), abs(delta.max()))
    vmax = max(vmax, 0.01)

    im = ax.imshow(delta, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(c) for c in col_labels], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Coefficient", fontsize=11)
    ax.set_title(f"Sensitivity Heatmap: Best F1 − Baseline ({baseline_f1:.3f})\n"
                 "(green = improves over baseline, red = degrades)",
                 pad=10, fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if not np.isnan(delta[i, j]):
                ax.text(j, i, f"{delta[i, j]:+.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(delta[i, j]) > vmax * 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("Δ F1 vs. baseline", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] sensitivity_heatmap.png")


# ── 4. Convergence speed: iterations to reach 90% of best F1 ─────────────────
def plot_convergence(results_dir: Path, out_dir: Path,
                     eval_interval: int = 10, smooth_sigma: int = 2):
    all_dirs = list(results_dir.glob("*__*"))
    variants = sorted(set(d.name.split("__")[0] for d in all_dirs))
    groups = group_variants(variants)

    baseline_dir = results_dir / "baseline__TwinkleTwinkleRousseau"
    b_hist = load_history(baseline_dir)
    b_f1   = np.array([h.get("f1", 0) for h in b_hist])

    # Threshold: 90% of smoothed baseline peak
    b_smooth = smooth(b_f1, smooth_sigma)
    threshold = 0.9 * b_smooth.max()

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(3.6 * n_groups, 3.5), squeeze=False)

    for col, (group, mvs) in enumerate(sorted(groups.items())):
        ax = axes[0, col]
        color = PALETTE.get(group, "#333333")
        mvs_sorted = sorted(mvs, key=parse_coef)

        for mv in mvs_sorted:
            coef = parse_coef(mv)
            ls, mk = COEF_LINESTYLE.get(coef, ("-", "o"))

            run_dir = results_dir / f"{mv}__TwinkleTwinkleRousseau"
            hist = load_history(run_dir)
            if not hist:
                continue
            f1 = np.array([h.get("f1", 0) for h in hist])
            f1_s = smooth(f1, smooth_sigma)
            best = f1_s.max()
            thresh = 0.9 * best

            # First index where smoothed F1 >= 90% of its own best
            mask = f1_s >= thresh
            idx = np.argmax(mask) if mask.any() else len(f1_s)
            real_iter = idx * eval_interval

            ax.bar(str(coef), real_iter, color=color, alpha=0.7, edgecolor="white")

        ax.set_xlabel("Coefficient", fontsize=10)
        ax.set_ylabel("Iters to 90% best F1", fontsize=10)
        ax.set_title(group.replace("_", " ").title(),
                     pad=8, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Convergence Speed: Iterations to Reach 90% of Best F1",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_convergence.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] sensitivity_convergence.png")


# ── 5. Combined summary table ─────────────────────────────────────────────────
def write_summary_table(results_dir: Path, out_dir: Path):
    all_dirs = list(results_dir.glob("*__*"))
    variants = sorted(set(d.name.split("__")[0] for d in all_dirs))
    groups = group_variants(variants)

    baseline_dir = results_dir / "baseline__TwinkleTwinkleRousseau"
    b_hist = load_history(baseline_dir)
    baseline_f1 = max((h.get("f1", 0) for h in b_hist), default=0)

    rows = []
    for group, mvs in sorted(groups.items()):
        for mv in sorted(mvs, key=parse_coef):
            run_dir = results_dir / f"{mv}__TwinkleTwinkleRousseau"
            hist = load_history(run_dir)
            if not hist:
                continue
            best_f1 = max((h.get("f1", 0) for h in hist))
            delta   = best_f1 - baseline_f1
            rows.append({
                "group": group,
                "coef": parse_coef(mv),
                "best_f1": best_f1,
                "delta_vs_baseline": delta,
                "improved": delta > 0,
            })

    # Pick best per group
    best_per_group = {}
    for r in rows:
        g = r["group"]
        if g not in best_per_group or r["best_f1"] > best_per_group[g]["best_f1"]:
            best_per_group[g] = r

    # Markdown table
    lines = [
        "# Sensitivity Analysis Summary\n",
        f"Baseline F1: **{baseline_f1:.4f}**\n",
        "| Group | Coef | Best F1 | Δ vs Baseline | Improved? |",
        "|---|---|---:|---:|---|",
    ]
    for r in rows:
        flag = "✓" if r["improved"] else "✗"
        delta_str = f"{r['delta_vs_baseline']:+.4f}"
        lines.append(
            f"| {r['group']} | {r['coef']} | {r['best_f1']:.4f} | "
            f"{delta_str} | {flag} |"
        )

    lines += [
        "",
        "## Best Coefficient per Group\n",
        "| Group | Best Coef | Best F1 | Δ vs Baseline |",
        "|---|---|---:|---:|",
    ]
    for g in sorted(best_per_group):
        r = best_per_group[g]
        lines.append(
            f"| {g} | {r['coef']} | {r['best_f1']:.4f} | "
            f"{r['delta_vs_baseline']:+.4f} |"
        )

    txt = "\n".join(lines) + "\n"
    (out_dir / "sensitivity_summary.md").write_text(txt)
    print("[OK] sensitivity_summary.md")

    # JSON for downstream use
    summary_json = {
        "baseline_f1": baseline_f1,
        "best_per_group": {g: {
            "coef": r["coef"],
            "best_f1": r["best_f1"],
            "delta": r["delta_vs_baseline"],
        } for g, r in best_per_group.items()},
        "all_rows": rows,
    }
    (out_dir / "sensitivity_summary.json").write_text(json.dumps(summary_json, indent=2))
    print("[OK] sensitivity_summary.json")

    return best_per_group, baseline_f1


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/Users/Sining/Desktop/Files/Course/Homework/2026春季/深度强化学习/Project/pianomime/results/reward_sensitivity"))
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--smooth-sigma", type=int, default=2,
                    help="Gaussian smoothing sigma for training curves")
    args = ap.parse_args()

    out_dir = args.output_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_training_curves(args.results_dir, out_dir,
                        eval_interval=args.eval_interval, smooth_sigma=args.smooth_sigma)
    plot_best_f1_bars(args.results_dir, out_dir)
    plot_f1_heatmap(args.results_dir, out_dir)
    plot_convergence(args.results_dir, out_dir, eval_interval=args.eval_interval,
                     smooth_sigma=args.smooth_sigma)
    best_per_group, baseline_f1 = write_summary_table(args.results_dir, out_dir)

    print(f"\n=== Best Coefficients ===")
    print(f"Baseline F1: {baseline_f1:.4f}\n")
    for g in sorted(best_per_group):
        r = best_per_group[g]
        flag = "✓ improved" if r["delta_vs_baseline"] > 0 else "✗ no improvement"
        print(f"  {g:<14}  coef={r['coef']:>5}  F1={r['best_f1']:.4f}  "
              f"{flag} ({r['delta_vs_baseline']:+.4f})")


if __name__ == "__main__":
    main()
