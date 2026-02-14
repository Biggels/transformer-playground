from __future__ import annotations

from torch import nn

from transformer_playground.model.attention import CausalSelfAttention


def build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        activation: str,
        mlp_enabled: bool,
        use_rope: bool,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_enabled = mlp_enabled
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            build_activation(activation),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.mlp_enabled:
            x = x + self.mlp(self.ln2(x))
        return x
