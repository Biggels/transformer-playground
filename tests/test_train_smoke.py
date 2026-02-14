from pathlib import Path

from transformer_playground.config import ExperimentConfig
from transformer_playground.train import run_training


def test_train_smoke(tmp_path: Path):
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("hello world\nhello there\nsmall test\nquick run\n", encoding="utf-8")

    cfg = ExperimentConfig()
    cfg.data.phrases_path = str(phrases)
    cfg.tracking.runs_dir = str(tmp_path / "runs")
    cfg.runtime.device = "cpu"

    cfg.tokenizer.vocab_size = 32
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.context_length = 8

    cfg.train.batch_size = 4
    cfg.train.grad_accum_steps = 1
    cfg.train.max_steps = 3
    cfg.train.eval_interval = 1
    cfg.train.eval_batches = 1
    cfg.train.log_interval = 1
    cfg.train.sample_interval = 1

    run_dir = run_training(cfg)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "samples.jsonl").exists()
    assert (run_dir / "checkpoints" / "best_val.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
