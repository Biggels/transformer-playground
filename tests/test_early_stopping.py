import json
from pathlib import Path

from transformer_playground.config import ExperimentConfig
from transformer_playground.train import run_training


def test_early_stopping_triggers_before_max_steps(tmp_path: Path):
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("alpha beta\nbeta gamma\ngamma delta\ndelta alpha\n", encoding="utf-8")

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
    cfg.train.max_steps = 20
    cfg.train.eval_interval = 1
    cfg.train.eval_batches = 1
    cfg.train.log_interval = 1
    cfg.train.sample_interval = 50

    cfg.train.early_stopping = True
    cfg.train.early_stopping_patience = 1
    cfg.train.early_stopping_min_delta = 10.0
    cfg.train.early_stopping_min_steps = 0

    cfg.tracking.auto_report = False
    cfg.tracking.auto_plot_loss = False

    run_dir = run_training(cfg)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["early_stopped"] is True
    assert summary["final_step"] < (cfg.train.max_steps - 1)
    assert summary["best_step"] == 0
