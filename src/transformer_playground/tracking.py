from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def make_run_dir(base_dir: str | Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
