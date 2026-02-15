from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from transformer_playground.config import ModelConfig
from transformer_playground.model.block import TransformerBlock
from transformer_playground.model.embeddings import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding


class DecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)

        if cfg.positional_encoding == "learned":
            self.position = LearnedPositionalEmbedding(cfg.context_length, cfg.d_model)
            use_rope = False
        elif cfg.positional_encoding == "sinusoidal":
            self.position = SinusoidalPositionalEmbedding(cfg.context_length, cfg.d_model)
            use_rope = False
        elif cfg.positional_encoding == "rope":
            self.position = nn.Identity()
            use_rope = True
        else:
            raise ValueError(f"Unknown positional encoding: {cfg.positional_encoding}")

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    activation=cfg.activation,
                    mlp_enabled=cfg.mlp_enabled,
                    use_rope=use_rope,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.token_embedding(idx)
        x = self.position(x)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_probs[mask] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, sampled)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token[finished] = eos_token_id
            idx = torch.cat((idx, next_token), dim=1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(1) == eos_token_id)
                if bool(finished.all()):
                    break
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
