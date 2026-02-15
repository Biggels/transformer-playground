from __future__ import annotations

import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from transformer_playground.data import PackedDataset, encode_phrases, read_phrases, split_phrases
from transformer_playground.infer import load_artifacts
from transformer_playground.train import evaluate
from transformer_playground.tracking import save_json


def _distinct_ratio(items: list[str], n: int) -> float:
    grams: list[tuple[str, ...]] = []
    total = 0
    for text in items:
        toks = [t for t in text.split() if t]
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            grams.append(tuple(toks[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(set(grams)) / total


def _generate_samples(
    model,
    tokenizer,
    device: str,
    prompt: str | None,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    out: list[str] = []
    seed_ids = tokenizer.encode(prompt) if prompt else [tokenizer.eos_id]

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            x = torch.tensor([seed_ids], dtype=torch.long, device=device)
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_id,
            )
            ids = y[0].tolist()
            if prompt:
                text = tokenizer.decode(ids).strip()
            else:
                completion = ids[len(seed_ids) :]
                if completion and completion[-1] == tokenizer.eos_id:
                    completion = completion[:-1]
                text = tokenizer.decode(completion).strip()
            out.append(text)
    return out


def build_report(
    run_path: str | Path,
    prompt: str | None,
    n_samples: int,
    max_new_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    checkpoint_name: str,
    eval_batches: int,
    save: bool = True,
) -> tuple[dict[str, object], Path | None]:
    run = Path(run_path)
    cfg, model, tokenizer, device = load_artifacts(run, checkpoint_name=checkpoint_name)
    if max_new_tokens is None:
        max_new_tokens = cfg.sampling.max_new_tokens
    if temperature is None:
        temperature = cfg.sampling.temperature
    if top_p is None:
        top_p = cfg.sampling.top_p

    all_phrases = read_phrases(cfg.data)
    train_phrases, val_phrases = split_phrases(all_phrases, cfg.data.val_fraction, cfg.data.seed)
    train_set = {p.strip() for p in train_phrases}
    val_set = {p.strip() for p in val_phrases}
    all_set = train_set | val_set

    train_tokens = encode_phrases(train_phrases, tokenizer, cfg.data.eos_text)
    val_tokens = encode_phrases(val_phrases, tokenizer, cfg.data.eos_text)
    dataset = PackedDataset(train_tokens, val_tokens, cfg.model)

    prev_eval_batches = cfg.train.eval_batches
    cfg.train.eval_batches = eval_batches
    losses = evaluate(model, dataset, cfg, device)
    cfg.train.eval_batches = prev_eval_batches

    samples = _generate_samples(
        model,
        tokenizer,
        device,
        prompt=prompt,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    cleaned = [s.strip() for s in samples]
    non_empty = [s for s in cleaned if s]
    unique = set(non_empty)

    match_train = [s for s in non_empty if s in train_set]
    match_val = [s for s in non_empty if s in val_set]
    match_any = [s for s in non_empty if s in all_set]

    token_lens = [len(tokenizer.encode(s)) for s in non_empty] if non_empty else [0]
    char_lens = [len(s) for s in non_empty] if non_empty else [0]

    counts = Counter(non_empty)

    report: dict[str, object] = {
        "run": str(run),
        "checkpoint": checkpoint_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation": {
            "prompt": prompt,
            "n_samples": n_samples,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "eval": losses,
        "metrics": {
            "non_empty_ratio": (len(non_empty) / n_samples) if n_samples else 0.0,
            "unique_ratio": (len(unique) / len(non_empty)) if non_empty else 0.0,
            "exact_match_train_ratio": (len(match_train) / len(non_empty)) if non_empty else 0.0,
            "exact_match_val_ratio": (len(match_val) / len(non_empty)) if non_empty else 0.0,
            "exact_match_any_ratio": (len(match_any) / len(non_empty)) if non_empty else 0.0,
            "novel_ratio": 1.0 - ((len(match_any) / len(non_empty)) if non_empty else 0.0),
            "distinct_1": _distinct_ratio(non_empty, 1),
            "distinct_2": _distinct_ratio(non_empty, 2),
            "avg_token_length": statistics.mean(token_lens),
            "avg_char_length": statistics.mean(char_lens),
        },
        "top_duplicates": counts.most_common(20),
        "samples": samples,
    }

    out_path: Path | None = None
    if save:
        reports_dir = run / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        suffix = checkpoint_name.replace(".pt", "")
        out_path = reports_dir / f"report-{suffix}-{int(time.time())}.json"
        save_json(out_path, report)

    return report, out_path
