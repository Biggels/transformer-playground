import torch

from transformer_playground.config import ModelConfig
from transformer_playground.model.decoder_lm import DecoderLM


def _run(cfg: ModelConfig):
    model = DecoderLM(cfg, vocab_size=128)
    x = torch.randint(0, 128, (2, cfg.context_length))
    y = torch.randint(0, 128, (2, cfg.context_length))
    logits, loss = model(x, y)
    assert logits.shape == (2, cfg.context_length, 128)
    assert loss is not None


def test_model_variants_forward():
    for pe in ("learned", "sinusoidal", "rope"):
        cfg = ModelConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_ratio=2.0,
            context_length=16,
            positional_encoding=pe,
            activation="gelu",
            mlp_enabled=True,
        )
        _run(cfg)


def test_model_without_mlp():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2.0,
        context_length=16,
        positional_encoding="learned",
        activation="relu",
        mlp_enabled=False,
    )
    _run(cfg)
