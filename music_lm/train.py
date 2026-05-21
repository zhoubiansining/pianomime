from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import trange

from music_lm.model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT on tokenized MAESTRO.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_tokens(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(tokens: np.ndarray, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - block_size - 1
    if max_start <= 0:
        raise ValueError("Token file is shorter than block_size + 1")
    starts = torch.randint(max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(np.asarray(tokens[i : i + block_size], dtype=np.int64)) for i in starts]
    )
    y = torch.stack(
        [torch.from_numpy(np.asarray(tokens[i + 1 : i + 1 + block_size], dtype=np.int64)) for i in starts]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, tokens in [("train", train_tokens), ("validation", val_tokens)]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = get_batch(tokens, batch_size, block_size, device)
            _, loss = model(x, y)
            assert loss is not None
            split_losses.append(loss.item())
        mean_loss = float(np.mean(split_losses))
        losses[split] = mean_loss
        losses[f"{split}_ppl"] = float(math.exp(min(20.0, mean_loss)))
    model.train()
    return losses


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tokens = load_tokens(args.data_dir / "train.bin")
    val_tokens = load_tokens(args.data_dir / "validation.bin")
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    vocab_size = int(manifest.get("vocab_size", int(max(train_tokens.max(), val_tokens.max())) + 1))

    device = torch.device(args.device)
    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    log_path = args.out_dir / "train_log.jsonl"
    progress = trange(1, args.max_steps + 1, desc="Training music GPT")
    for step in progress:
        x, y = get_batch(train_tokens, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.4f}")
        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            losses = estimate_loss(
                model,
                train_tokens,
                val_tokens,
                args.batch_size,
                args.block_size,
                args.eval_iters,
                device,
            )
            record = {"step": step, "loss": loss.item(), **losses}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            is_best = losses["validation"] < best_val
            if is_best:
                best_val = losses["validation"]
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model_config.to_dict(),
                "tokenizer_config": manifest.get("tokenizer", {}),
                "step": step,
                "best_validation_loss": best_val,
                "args": vars(args),
            }
            torch.save(checkpoint, args.out_dir / "latest.pt")
            if is_best:
                torch.save(checkpoint, args.out_dir / "best.pt")
            print(json.dumps(record))


if __name__ == "__main__":
    main()
