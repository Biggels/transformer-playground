from pathlib import Path

from transformer_playground.config import ExperimentConfig
from transformer_playground.train import run_training


def test_auto_report_and_plot_generated(tmp_path: Path):
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
    cfg.train.max_steps = 2
    cfg.train.eval_interval = 1
    cfg.train.eval_batches = 1
    cfg.train.log_interval = 1
    cfg.train.sample_interval = 1

    cfg.tracking.auto_plot_loss = True
    cfg.tracking.auto_plot_log_y = True
    cfg.tracking.auto_report = True
    cfg.tracking.auto_report_n_samples = 4
    cfg.tracking.auto_report_eval_batches = 1
    cfg.tracking.auto_report_unconditional = True

    run_dir = run_training(cfg)

    assert (run_dir / "post_run_artifacts.json").exists()
    reports_dir = run_dir / "reports"
    assert any(p.name.startswith("report-") and p.suffix == ".json" for p in reports_dir.iterdir())
    assert any(p.name.startswith("loss-curve-") and p.suffix == ".svg" for p in reports_dir.iterdir())
