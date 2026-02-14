from pathlib import Path

from transformer_playground.config import DataConfig
from transformer_playground.data import read_phrases, split_phrases


def test_read_phrases_strips_empty(tmp_path: Path):
    p = tmp_path / "phrases.txt"
    p.write_text("A phrase\n\n  Another one  \n", encoding="utf-8")
    cfg = DataConfig(phrases_path=str(p))
    out = read_phrases(cfg)
    assert out == ["A phrase", "Another one"]


def test_split_phrases_deterministic():
    phrases = [f"p{i}" for i in range(100)]
    a_train, a_val = split_phrases(phrases, 0.1, 42)
    b_train, b_val = split_phrases(phrases, 0.1, 42)
    assert a_train == b_train
    assert a_val == b_val
    assert len(a_val) == 10
