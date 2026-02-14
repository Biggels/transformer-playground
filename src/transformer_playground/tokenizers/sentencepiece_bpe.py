from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from transformer_playground.config import TokenizerConfig
from transformer_playground.tokenizers.base import Tokenizer


class SentencePieceBPETokenizer(Tokenizer):
    def __init__(self, cfg: TokenizerConfig) -> None:
        self.cfg = cfg
        self.processor = spm.SentencePieceProcessor()
        self._model_path: Path | None = None

    def train(self, corpus_path: str | Path, out_dir: str | Path, eos_text: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = out / "spm"
        model_path = prefix.with_suffix(".model")
        if model_path.exists() and self.cfg.train_if_missing:
            self.load(model_path)
            return
        user_symbols = eos_text
        cmd = (
            f"--input={Path(corpus_path)} "
            f"--model_prefix={prefix} "
            f"--model_type={self.cfg.model_type} "
            f"--vocab_size={self.cfg.vocab_size} "
            "--bos_id=-1 --pad_id=-1 --unk_id=0 "
            f"--user_defined_symbols={user_symbols} "
            f"--character_coverage={self.cfg.character_coverage}"
        )
        spm.SentencePieceTrainer.Train(cmd)
        self.load(model_path)

    def load(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self._model_path))

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, token_ids: list[int]) -> str:
        return self.processor.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    @property
    def eos_id(self) -> int:
        eos_id = self.processor.piece_to_id("<eos>")
        if eos_id < 0:
            raise ValueError("Tokenizer does not define <eos> token")
        return int(eos_id)
