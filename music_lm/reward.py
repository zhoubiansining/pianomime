from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from dm_env_wrappers import EnvironmentWrapper
except ModuleNotFoundError:
    class EnvironmentWrapper:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("dm_env_wrappers is required for MusicPPLRewardWrapper")

from music_lm.model import GPT, GPTConfig
from music_lm.tokenizer import EventTokenizer, TokenizerConfig


class MusicPerplexityEvaluator:
    """Loads a trained music GPT checkpoint and scores MIDI/token sequences."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        tokenizer_config = TokenizerConfig.from_dict(checkpoint.get("tokenizer_config", {}))
        self.tokenizer = EventTokenizer(tokenizer_config)
        self.model_config = GPTConfig.from_dict(checkpoint["model_config"])
        self.model = GPT(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def nll(self, tokens: Sequence[int]) -> tuple[float, int]:
        tokens = [int(token) for token in tokens]
        if len(tokens) < 2:
            return 0.0, 0
        total_nll = 0.0
        total_count = 0
        block = self.model_config.block_size
        stride = block
        for start in range(0, len(tokens) - 1, stride):
            chunk = tokens[start : start + block + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long, device=self.device).unsqueeze(0)
            logits, _ = self.model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss.item())
            total_count += int(y.numel())
        return total_nll, total_count

    def log_perplexity(self, tokens: Sequence[int]) -> float:
        total_nll, count = self.nll(tokens)
        if count == 0:
            return float("nan")
        return total_nll / count

    def perplexity(self, tokens: Sequence[int]) -> float:
        log_ppl = self.log_perplexity(tokens)
        if math.isnan(log_ppl):
            return float("nan")
        return float(math.exp(min(20.0, log_ppl)))

    def score_midi_file(self, path: str | Path) -> dict[str, float]:
        tokens = self.tokenizer.encode_midi_file(path)
        log_ppl = self.log_perplexity(tokens)
        return {
            "tokens": len(tokens),
            "log_ppl": log_ppl,
            "ppl": float(math.exp(min(20.0, log_ppl))),
        }

    def score_midi_messages(self, messages: Sequence[object]) -> dict[str, float]:
        tokens = self.tokenizer.encode_midi_messages(messages, add_bos=True, add_eos=True)
        log_ppl = self.log_perplexity(tokens)
        return {
            "tokens": len(tokens),
            "log_ppl": log_ppl,
            "ppl": float(math.exp(min(20.0, log_ppl))),
        }


class MusicPPLRewardWrapper(EnvironmentWrapper):
    """Adds an optional frozen music-LM perplexity bonus to environment reward.

    The wrapper is intentionally conservative: it only evaluates when new MIDI
    events appear, uses a bounded sliding-window reward, and also exposes episode
    PPL metrics for logging/evaluation.
    """

    def __init__(
        self,
        environment: object,
        checkpoint_path: str | Path,
        reward_weight: float = 0.0,
        window_tokens: int = 256,
        reward_clip: float = 5.0,
        reference_log_ppl: float | None = None,
        min_tokens: int = 8,
        deque_size: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(environment)
        self.evaluator = MusicPerplexityEvaluator(checkpoint_path, device=device)
        self.reward_weight = float(reward_weight)
        self.window_tokens = int(window_tokens)
        self.reward_clip = float(reward_clip)
        self.reference_log_ppl = reference_log_ppl
        self.min_tokens = int(min_tokens)
        self._tokens: list[int] = []
        self._message_count = 0
        self._episode_log_ppls = deque(maxlen=deque_size)
        self._episode_ppls = deque(maxlen=deque_size)

    def reset(self) -> dm_env.TimeStep:
        self._tokens = [self.evaluator.tokenizer.bos_token]
        self._message_count = 0
        return self._environment.reset()

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        new_tokens = self._latest_tokens()
        self._tokens.extend(new_tokens)

        bonus = 0.0
        if new_tokens and len(self._tokens) >= self.min_tokens:
            window = self._tokens[-self.window_tokens :]
            log_ppl = self.evaluator.log_perplexity(window)
            if not math.isnan(log_ppl):
                center = self.reference_log_ppl if self.reference_log_ppl is not None else 0.0
                bonus = -self.reward_weight * (log_ppl - center)
                if self.reward_clip > 0:
                    bonus = float(np.clip(bonus, -self.reward_clip, self.reward_clip))

        if timestep.reward is not None and bonus:
            timestep = timestep._replace(reward=float(timestep.reward) + bonus)

        if timestep.last():
            episode_tokens = self._tokens + [self.evaluator.tokenizer.eos_token]
            log_ppl = self.evaluator.log_perplexity(episode_tokens)
            if not math.isnan(log_ppl):
                self._episode_log_ppls.append(log_ppl)
                self._episode_ppls.append(float(math.exp(min(20.0, log_ppl))))
        return timestep

    def get_music_lm_metrics(self) -> dict[str, float]:
        if not self._episode_log_ppls:
            raise ValueError("No music-LM episode metrics available yet.")
        return {
            "music_lm_log_ppl": float(np.mean(self._episode_log_ppls)),
            "music_lm_ppl": float(np.mean(self._episode_ppls)),
        }

    def _latest_tokens(self) -> list[int]:
        try:
            task = self._task()
            messages = task.piano.midi_module.get_all_midi_messages()
        except Exception:
            return []
        new_messages = messages[self._message_count :]
        self._message_count = len(messages)
        return self.evaluator.tokenizer.encode_midi_messages(
            new_messages,
            add_bos=False,
            add_eos=False,
        )

    def _task(self):
        env = self._environment
        for _ in range(20):
            task = getattr(env, "task", None)
            if task is not None:
                return task
            env = getattr(env, "_environment", None)
            if env is None:
                break
        raise AttributeError("Could not find wrapped RoboPianist task.")
