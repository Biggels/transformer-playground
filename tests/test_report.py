from pathlib import Path

from transformer_playground.config import ExperimentConfig
from transformer_playground.report import build_report
from transformer_playground.train import run_training


def _make_run(tmp_path: Path) -> Path:
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

    return run_training(cfg)


def test_build_report(tmp_path: Path):
    run_dir = _make_run(tmp_path)
    report, out_path = build_report(
        run_path=run_dir,
        prompt=None,
        n_samples=5,
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        checkpoint_name="best_val.pt",
        eval_batches=1,
        save=True,
    )
    assert "metrics" in report
    assert "eval" in report
    assert len(report["samples"]) == 5
    assert out_path is not None
    assert out_path.exists()
