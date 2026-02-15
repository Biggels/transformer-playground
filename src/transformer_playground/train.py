from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch

from transformer_playground.config import ExperimentConfig, config_to_dict, ensure_device, save_config_json
from transformer_playground.data import PackedDataset, encode_phrases, read_phrases, split_phrases
from transformer_playground.model.decoder_lm import DecoderLM, count_parameters
from transformer_playground.tokenizers.sentencepiece_bpe import SentencePieceBPETokenizer
from transformer_playground.tracking import append_jsonl, make_run_dir, save_json


def build_tokenizer(cfg: ExperimentConfig, tokenizer_dir: Path) -> SentencePieceBPETokenizer:
    if cfg.tokenizer.kind != "sentencepiece_bpe":
        raise ValueError(f"Unsupported tokenizer kind: {cfg.tokenizer.kind}")
    tok = SentencePieceBPETokenizer(cfg.tokenizer)
    tok.train(cfg.data.phrases_path, tokenizer_dir, cfg.data.eos_text)
    return tok


def get_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = max_lr * min_lr_ratio
    return min_lr + cosine * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model: DecoderLM, dataset: PackedDataset, cfg: ExperimentConfig, device: str) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(cfg.train.eval_batches):
            xb, yb = dataset.get_batch(split, cfg.train.batch_size, device)
            _, loss = model(xb, yb)
            losses.append(float(loss.item()))
        mean_loss = sum(losses) / len(losses)
        out[f"{split}_loss"] = mean_loss
        out[f"{split}_ppl"] = math.exp(mean_loss)
    model.train()
    return out


@torch.no_grad()
def sample_text(model: DecoderLM, tokenizer: SentencePieceBPETokenizer, cfg: ExperimentConfig, device: str) -> str:
    model.eval()
    prompt = "a"
    inp = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        inp,
        max_new_tokens=cfg.sampling.max_new_tokens,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        eos_token_id=tokenizer.eos_id,
    )
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text


def save_checkpoint(
    path: Path,
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
        },
        path,
    )


def run_training(cfg: ExperimentConfig) -> Path:
    torch.manual_seed(cfg.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.data.seed)

    run_dir = make_run_dir(cfg.tracking.runs_dir)
    save_config_json(cfg, run_dir / "config.json")

    tokenizer = build_tokenizer(cfg, run_dir / "tokenizer")
    phrases = read_phrases(cfg.data)
    train_phrases, val_phrases = split_phrases(phrases, cfg.data.val_fraction, cfg.data.seed)

    train_tokens = encode_phrases(train_phrases, tokenizer, cfg.data.eos_text)
    val_tokens = encode_phrases(val_phrases, tokenizer, cfg.data.eos_text)

    device = ensure_device(cfg)
    model = DecoderLM(cfg.model, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device == "cuda")
    dataset = PackedDataset(train_tokens, val_tokens, cfg.model)

    best_val = float("inf")
    t0 = time.time()
    total_tokens = 0

    for step in range(cfg.train.max_steps):
        lr = get_lr(step, cfg.train.max_steps, cfg.train.warmup_steps, cfg.train.lr, cfg.train.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(cfg.train.grad_accum_steps):
            xb, yb = dataset.get_batch("train", cfg.train.batch_size, device)
            total_tokens += xb.numel()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler.is_enabled()):
                _, loss = model(xb, yb)
                loss = loss / cfg.train.grad_accum_steps
            loss_total += float(loss.item())
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.train.log_interval == 0:
            elapsed = max(1e-6, time.time() - t0)
            tokens_per_sec = total_tokens / elapsed
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "step": step,
                    "train_loss": loss_total,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                },
            )

        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps - 1:
            stats = evaluate(model, dataset, cfg, device)
            payload: dict[str, Any] = {"step": step, **stats, "lr": lr}
            append_jsonl(run_dir / "metrics.jsonl", payload)
            if stats["val_loss"] < best_val:
                best_val = stats["val_loss"]
                save_checkpoint(run_dir / "checkpoints" / "best_val.pt", model, optimizer, scaler, step)

        if step % cfg.train.sample_interval == 0 or step == cfg.train.max_steps - 1:
            txt = sample_text(model, tokenizer, cfg, device)
            append_jsonl(run_dir / "samples.jsonl", {"step": step, "text": txt})

    save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scaler, cfg.train.max_steps - 1)
    save_json(
        run_dir / "summary.json",
        {
            "params": count_parameters(model),
            "best_val_loss": best_val,
            "config": config_to_dict(cfg),
            "run_dir": str(run_dir),
        },
    )

    return run_dir
