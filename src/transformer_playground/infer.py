from __future__ import annotations

import json
from pathlib import Path

import torch

from transformer_playground.config import ExperimentConfig, ensure_device, load_config_json
from transformer_playground.model.decoder_lm import DecoderLM
from transformer_playground.tokenizers.sentencepiece_bpe import SentencePieceBPETokenizer


def resolve_run_path(runs_dir: str | Path, run_id: str) -> Path:
    p = Path(runs_dir) / run_id
    if not p.exists():
        raise FileNotFoundError(f"Run not found: {p}")
    return p


def load_artifacts(run_path: str | Path, checkpoint_name: str = "best_val.pt") -> tuple[ExperimentConfig, DecoderLM, SentencePieceBPETokenizer, str]:
    run = Path(run_path)
    cfg = load_config_json(run / "config.json")
    device = ensure_device(cfg)

    tokenizer = SentencePieceBPETokenizer(cfg.tokenizer)
    tokenizer.load(run / "tokenizer" / "spm.model")

    model = DecoderLM(cfg.model, tokenizer.vocab_size).to(device)
    ckpt = torch.load(run / "checkpoints" / checkpoint_name, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model, tokenizer, device


def generate_text(
    run_path: str | Path,
    prompt: str | None,
    max_new_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    checkpoint_name: str = "best_val.pt",
) -> str:
    cfg, model, tokenizer, device = load_artifacts(run_path, checkpoint_name=checkpoint_name)
    if max_new_tokens is None:
        max_new_tokens = cfg.sampling.max_new_tokens
    if temperature is None:
        temperature = cfg.sampling.temperature
    if top_p is None:
        top_p = cfg.sampling.top_p
    seed_ids = tokenizer.encode(prompt) if prompt else [tokenizer.eos_id]
    x = torch.tensor([seed_ids], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_id,
    )
    out_ids = y[0].tolist()
    if not prompt:
        completion = out_ids[len(seed_ids) :]
        if completion and completion[-1] == tokenizer.eos_id:
            completion = completion[:-1]
        return tokenizer.decode(completion).strip()
    return tokenizer.decode(out_ids)


def compare_runs(
    run_paths: list[str | Path],
    prompt: str | None,
    n_samples: int,
    max_new_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    checkpoint_name: str = "best_val.pt",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in run_paths:
        outputs: list[str] = []
        for _ in range(n_samples):
            outputs.append(
                generate_text(
                    run,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    checkpoint_name=checkpoint_name,
                )
            )
        summary = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
        rows.append({"run": str(run), "params": summary.get("params"), "samples": outputs})
    return rows
