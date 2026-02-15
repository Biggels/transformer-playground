from __future__ import annotations

import dataclasses
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass
class DataConfig:
    phrases_path: str = "phrases.txt"
    val_fraction: float = 0.1
    seed: int = 1337
    preserve_case: bool = True
    strip_whitespace: bool = True
    eos_text: str = "<eos>"


@dataclass
class TokenizerConfig:
    kind: str = "sentencepiece_bpe"
    vocab_size: int = 3200
    character_coverage: float = 1.0
    model_type: str = "bpe"
    train_if_missing: bool = True


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    context_length: int = 64
    positional_encoding: str = "learned"  # learned | sinusoidal | rope
    mlp_enabled: bool = True
    activation: str = "gelu"  # gelu | relu | silu
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    grad_accum_steps: int = 2
    max_steps: int = 1200
    eval_interval: int = 100
    eval_batches: int = 32
    log_interval: int = 20
    sample_interval: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    amp: bool = True


@dataclass
class EvalConfig:
    random_split: str = "90_10"


@dataclass
class SamplingConfig:
    max_new_tokens: int = 24
    temperature: float = 0.9
    top_p: float = 0.9


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    num_workers: int = 0


@dataclass
class TrackingConfig:
    runs_dir: str = "runs"
    save_best_only: bool = False
    auto_plot_loss: bool = True
    auto_plot_log_y: bool = True
    auto_report: bool = True
    auto_report_n_samples: int = 200
    auto_report_eval_batches: int = 32
    auto_report_unconditional: bool = True
    auto_report_prompt: str = "a"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def save_config_json(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")


def _load_module(path: str | Path) -> ModuleType:
    p = Path(path)
    spec = importlib.util.spec_from_file_location("tp_user_config", p)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load config module: {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        return default_config()
    module = _load_module(path)
    if hasattr(module, "get_config"):
        cfg = module.get_config()
    elif hasattr(module, "CONFIG"):
        cfg = module.CONFIG
    else:
        raise ValueError("Config module must define get_config() or CONFIG")
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError("Config object must be an ExperimentConfig")
    return cfg


def _parse_like(value: str, like: Any) -> Any:
    if isinstance(like, bool):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        if val in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid bool: {value}")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, tuple):
        items = [x.strip() for x in value.split(",") if x.strip()]
        if len(items) != len(like):
            raise ValueError(f"Expected {len(like)} values for tuple")
        return tuple(_parse_like(v, like[i]) for i, v in enumerate(items))
    return value


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise AttributeError(f"Unknown config path: {key}")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise AttributeError(f"Unknown config path: {key}")
        current = getattr(obj, leaf)
        parsed = _parse_like(raw, current)
        setattr(obj, leaf, parsed)
    return cfg


def ensure_device(cfg: ExperimentConfig) -> str:
    import torch

    if cfg.runtime.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def from_dict(d: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(**d["data"]),
        tokenizer=TokenizerConfig(**d["tokenizer"]),
        model=ModelConfig(**d["model"]),
        train=TrainConfig(**d["train"]),
        eval=EvalConfig(**d.get("eval", {})),
        sampling=SamplingConfig(**d["sampling"]),
        runtime=RuntimeConfig(**d["runtime"]),
        tracking=TrackingConfig(**d["tracking"]),
    )


def load_config_json(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_dict(data)
