from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_lm.reward import MusicPerplexityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MIDI files with a trained music GPT.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("midi_files", nargs="+", type=Path)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = MusicPerplexityEvaluator(args.checkpoint, device=args.device)
    for midi_file in args.midi_files:
        result = evaluator.score_midi_file(midi_file)
        result["midi_file"] = str(midi_file)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
