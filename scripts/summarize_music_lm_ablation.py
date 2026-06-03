#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_metrics(path: Path) -> list[dict[str, float | int | None]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if value in (None, "", "None", "nan"):
                    parsed[key] = None
                    continue
                if key in ("iteration", "env_steps"):
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def best_by(rows: list[dict[str, float | int | None]], metric: str) -> dict[str, float | int | None]:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        raise ValueError(f"No {metric} values found")
    return max(candidates, key=lambda row: float(row[metric]))


def summarize(path: Path) -> dict:
    rows = read_metrics(path)
    final = rows[-1]
    best_f1 = best_by(rows, "f1")
    return {
        "path": str(path),
        "num_evals": len(rows),
        "final": final,
        "best_f1": best_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize no-LM vs Music-LM PPO ablation metrics.")
    parser.add_argument("no_lm_metrics", type=Path)
    parser.add_argument("with_lm_metrics", type=Path)
    args = parser.parse_args()

    no_lm = summarize(args.no_lm_metrics)
    with_lm = summarize(args.with_lm_metrics)
    result = {
        "no_lm": no_lm,
        "with_lm": with_lm,
        "delta_final_f1": with_lm["final"].get("f1") - no_lm["final"].get("f1"),
        "delta_best_f1": with_lm["best_f1"].get("f1") - no_lm["best_f1"].get("f1"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
