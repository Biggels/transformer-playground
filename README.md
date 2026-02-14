# transformer-playground

A configurable decoder-only transformer playground in pure PyTorch for small local experiments.

## Goals
- Fast local iteration on small datasets
- Architecture-level ablations from config (not code edits)
- Reproducible run artifacts (config, metrics, checkpoints, samples)

## Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Dataset
Default path is `phrases.txt` (one phrase per line).

## Train
```bash
tp train --config configs/default.py
```

Override config keys from CLI:
```bash
tp train --config configs/default.py --set model.mlp_enabled=false --set model.positional_encoding=rope
```

## Generate
```bash
tp generate --runs-dir runs --run-id <RUN_ID> --prompt "a" --max-new-tokens 24 --temperature 0.9 --top-p 0.9
```

## Compare runs
```bash
tp compare --runs-dir runs --run-ids <RUN1>,<RUN2> --prompt "a" --n-samples 3
```

## Run artifacts
Each training run writes:
- `runs/<run_id>/config.json`
- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/samples.jsonl`
- `runs/<run_id>/checkpoints/best_val.pt`
- `runs/<run_id>/checkpoints/last.pt`
- `runs/<run_id>/tokenizer/spm.model`
- `runs/<run_id>/summary.json`
