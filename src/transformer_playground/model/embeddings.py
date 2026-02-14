from __future__ import annotations

import math

import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(context_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        pos = torch.arange(t, device=x.device)
        pos_emb = self.embedding(pos)[None, :, :]
        return x + pos_emb


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        return x + self.pe[:t].unsqueeze(0)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, rope: RotaryEmbedding) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope.get_cos_sin(q.size(-2), q.device)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
