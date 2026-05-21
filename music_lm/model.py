from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, values: dict) -> "GPTConfig":
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :length, :length] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = idx.shape
        if length > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {length}; block_size={self.config.block_size}"
            )
        pos = torch.arange(0, length, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
        return logits, loss

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
