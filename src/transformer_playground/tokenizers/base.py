from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Tokenizer(ABC):
    @abstractmethod
    def train(self, corpus_path: str | Path, out_dir: str | Path, eos_text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_id(self) -> int:
        raise NotImplementedError
