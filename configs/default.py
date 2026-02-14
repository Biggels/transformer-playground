from transformer_playground.config import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.phrases_path = "phrases.txt"
    cfg.data.val_fraction = 0.1

    cfg.tokenizer.kind = "sentencepiece_bpe"
    cfg.tokenizer.vocab_size = 3200

    cfg.model.d_model = 384
    cfg.model.n_layers = 6
    cfg.model.n_heads = 6
    cfg.model.context_length = 64
    cfg.model.mlp_enabled = True
    cfg.model.positional_encoding = "learned"
    cfg.model.activation = "gelu"

    cfg.train.batch_size = 64
    cfg.train.grad_accum_steps = 2
    cfg.train.max_steps = 800
    cfg.train.eval_interval = 100
    cfg.train.sample_interval = 100
    cfg.train.log_interval = 20
    cfg.train.eval_batches = 16

    cfg.sampling.max_new_tokens = 24
    cfg.sampling.temperature = 0.9
    cfg.sampling.top_p = 0.9

    cfg.runtime.device = "cuda"
    cfg.tracking.runs_dir = "runs"
    return cfg
