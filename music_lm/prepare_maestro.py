from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm

from music_lm.tokenizer import EventTokenizer, TokenizerConfig


MAESTRO_MIDI_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0-midi.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize MAESTRO MIDI files for GPT pretraining.")
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download MAESTRO v3.0.0 MIDI-only zip first.")
    parser.add_argument("--url", default=MAESTRO_MIDI_URL)
    parser.add_argument("--limit-per-split", type=int, default=0, help="Debug limit; 0 means all files.")
    parser.add_argument("--time-step-seconds", type=float, default=0.01)
    parser.add_argument("--max-time-shift-steps", type=int, default=100)
    parser.add_argument("--velocity-bins", type=int, default=32)
    return parser.parse_args()


def maybe_download(root: Path, url: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / Path(url).name
    if not zip_path.exists():
        print(f"Downloading {url} -> {zip_path}")
        urlretrieve(url, zip_path)
    marker = root / ".maestro_extracted"
    if marker.exists():
        return
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    marker.write_text("ok\n")


def find_metadata_csv(root: Path) -> Path:
    candidates = sorted(root.rglob("maestro-v*.csv"))
    if not candidates:
        candidates = sorted(root.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No MAESTRO metadata CSV found under {root}")
    return candidates[0]


def write_split(out_path: Path, tokens: list[int]) -> None:
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(out_path)


def main() -> None:
    args = parse_args()
    if args.download:
        maybe_download(args.maestro_root, args.url)

    metadata_csv = find_metadata_csv(args.maestro_root)
    dataset_root = metadata_csv.parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = EventTokenizer(
        TokenizerConfig(
            time_step_seconds=args.time_step_seconds,
            max_time_shift_steps=args.max_time_shift_steps,
            velocity_bins=args.velocity_bins,
        )
    )

    split_tokens: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    split_counts = {key: 0 for key in split_tokens}
    failures: list[dict[str, str]] = []

    with metadata_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Tokenizing MAESTRO"):
        split = row.get("split", "train")
        if split not in split_tokens:
            continue
        if args.limit_per_split and split_counts[split] >= args.limit_per_split:
            continue
        midi_path = dataset_root / row["midi_filename"]
        try:
            tokens = tokenizer.encode_midi_file(midi_path)
        except Exception as exc:
            failures.append({"midi_filename": row["midi_filename"], "error": str(exc)})
            continue
        split_tokens[split].extend(tokens)
        split_counts[split] += 1

    for split, tokens in split_tokens.items():
        if not tokens:
            raise RuntimeError(f"No tokens were produced for split {split}")
        write_split(out_dir / f"{split}.bin", tokens)

    manifest = {
        "metadata_csv": str(metadata_csv),
        "tokenizer": tokenizer.to_config_dict(),
        "vocab_size": tokenizer.vocab_size,
        "split_counts": split_counts,
        "split_tokens": {split: len(tokens) for split, tokens in split_tokens.items()},
        "failures": failures[:50],
        "num_failures": len(failures),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
