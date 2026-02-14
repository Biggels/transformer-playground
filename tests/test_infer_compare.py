from pathlib import Path

from transformer_playground.config import ExperimentConfig
from transformer_playground.infer import compare_runs, generate_text
from transformer_playground.train import run_training


def _make_run(tmp_path: Path, seed: int) -> Path:
    phrases = tmp_path / f"phrases_{seed}.txt"
    phrases.write_text("alpha beta\nbeta gamma\ngamma delta\ndelta alpha\n", encoding="utf-8")

    cfg = ExperimentConfig()
    cfg.data.seed = seed
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


def test_generate_and_compare(tmp_path: Path):
    run1 = _make_run(tmp_path, 1)
    run2 = _make_run(tmp_path, 2)

    text = generate_text(run1, prompt="alpha", max_new_tokens=4, temperature=1.0, top_p=1.0)
    assert isinstance(text, str)
    assert len(text) > 0

    rows = compare_runs(
        [run1, run2],
        prompt="alpha",
        n_samples=1,
        max_new_tokens=3,
        temperature=1.0,
        top_p=1.0,
    )
    assert len(rows) == 2
    assert len(rows[0]["samples"]) == 1
