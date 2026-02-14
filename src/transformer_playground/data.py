from __future__ import annotations

import random
from pathlib import Path

import torch

from transformer_playground.config import DataConfig, ModelConfig
from transformer_playground.tokenizers.base import Tokenizer


def read_phrases(cfg: DataConfig) -> list[str]:
    lines = Path(cfg.phrases_path).read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        value = line.strip() if cfg.strip_whitespace else line
        if value:
            out.append(value)
    return out


def split_phrases(phrases: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    items = list(phrases)
    rng.shuffle(items)
    val_size = max(1, int(len(items) * val_fraction))
    val = items[:val_size]
    train = items[val_size:]
    return train, val


def encode_phrases(phrases: list[str], tokenizer: Tokenizer, eos_text: str) -> torch.Tensor:
    all_ids: list[int] = []
    for p in phrases:
        all_ids.extend(tokenizer.encode(p))
        all_ids.extend(tokenizer.encode(eos_text))
    return torch.tensor(all_ids, dtype=torch.long)


class PackedDataset:
    def __init__(self, train_tokens: torch.Tensor, val_tokens: torch.Tensor, model_cfg: ModelConfig) -> None:
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        self.block_size = model_cfg.context_length

    def _sample(self, source: torch.Tensor, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.block_size + 2
        if len(source) < needed:
            repeats = (needed + len(source) - 1) // len(source)
            source = source.repeat(repeats)

        max_start = len(source) - self.block_size - 1
        if max_start <= 0:
            raise ValueError("Token stream too short for selected context_length")
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([source[s : s + self.block_size] for s in starts])
        y = torch.stack([source[s + 1 : s + 1 + self.block_size] for s in starts])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def get_batch(self, split: str, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train_tokens if split == "train" else self.val_tokens
        return self._sample(source, batch_size, device)
