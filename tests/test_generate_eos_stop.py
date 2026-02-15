import torch

from transformer_playground.config import ModelConfig
from transformer_playground.model.decoder_lm import DecoderLM


def test_generate_stops_when_eos_sampled(monkeypatch):
    cfg = ModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        context_length=8,
        positional_encoding="learned",
    )
    model = DecoderLM(cfg, vocab_size=16)

    eos_id = 3

    def fake_forward(idx, targets=None):
        b, t = idx.shape
        logits = torch.zeros((b, t, 16), device=idx.device)
        logits[:, -1, eos_id] = 10.0
        return logits, None

    monkeypatch.setattr(model, "forward", fake_forward)

    x = torch.tensor([[1, 2]], dtype=torch.long)
    y = model.generate(x, max_new_tokens=5, temperature=0.0, eos_token_id=eos_id)
    assert y.shape[1] == x.shape[1] + 1
    assert y[0, -1].item() == eos_id


def test_generate_without_eos_runs_full_length(monkeypatch):
    cfg = ModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        context_length=8,
        positional_encoding="learned",
    )
    model = DecoderLM(cfg, vocab_size=16)

    def fake_forward(idx, targets=None):
        b, t = idx.shape
        logits = torch.zeros((b, t, 16), device=idx.device)
        logits[:, -1, 4] = 10.0
        return logits, None

    monkeypatch.setattr(model, "forward", fake_forward)

    x = torch.tensor([[1, 2]], dtype=torch.long)
    y = model.generate(x, max_new_tokens=5, temperature=0.0, eos_token_id=None)
    assert y.shape[1] == x.shape[1] + 5
