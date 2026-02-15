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
If you omit sampling flags, command uses the sampling values saved in that run's `config.json`.

Unconditional phrase sampling (no textual prompt):
```bash
tp generate --runs-dir runs --run-id <RUN_ID> --unconditional --max-new-tokens 24 --temperature 0.9 --top-p 0.9
```

## Compare runs
```bash
tp compare --runs-dir runs --run-ids <RUN1>,<RUN2> --prompt "a" --n-samples 3
```

## Report a checkpoint
Generate a quality/overfitting report with eval stats and sampled-text metrics:
```bash
tp report --runs-dir runs --run-id <RUN_ID> --unconditional --n-samples 200 --max-new-tokens 24 --temperature 0.9 --top-p 0.9
```
If omitted, sampling flags default to the checkpoint run's saved config.

The command prints a summary and saves full JSON to:
- `runs/<run_id>/reports/report-<checkpoint>-<timestamp>.json`

## Plot train/val loss
Create an SVG curve from `metrics.jsonl`:
```bash
tp plot-loss --runs-dir runs --run-id <RUN_ID>
```

Options:
- `--out <path>` custom output path
- `--log-y` use log scale on y-axis (useful when val loss explodes)

## Run artifacts
Each training run writes:
- `runs/<run_id>/config.json`
- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/samples.jsonl`
- `runs/<run_id>/checkpoints/best_val.pt`
- `runs/<run_id>/checkpoints/last.pt`
- `runs/<run_id>/tokenizer/spm.model`
- `runs/<run_id>/summary.json`
