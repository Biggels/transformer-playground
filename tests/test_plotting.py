from pathlib import Path

from transformer_playground.plotting import load_loss_points


def test_load_loss_points(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                '{"step":0,"train_loss":5.0,"lr":0.001}',
                '{"step":0,"train_loss":4.8,"val_loss":5.2}',
                '{"step":10,"train_loss":4.1,"lr":0.001}',
                '{"step":10,"train_loss":4.0,"val_loss":4.9}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    t_steps, t_losses, v_steps, v_losses = load_loss_points(metrics)
    assert t_steps == [0, 10]
    assert t_losses == [5.0, 4.1]
    assert v_steps == [0, 10]
    assert v_losses == [5.2, 4.9]
