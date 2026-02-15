from transformer_playground.config import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.phrases_path = "phrases.txt"
    cfg.data.val_fraction = 0.1

    cfg.tokenizer.kind = "sentencepiece_bpe"
    cfg.tokenizer.vocab_size = 1600

    cfg.model.d_model = 256
    cfg.model.n_layers = 6
    cfg.model.n_heads = 8
    cfg.model.context_length = 64
    cfg.model.dropout = 0.2
    cfg.model.mlp_enabled = True
    cfg.model.positional_encoding = "learned"
    cfg.model.activation = "gelu"

    cfg.train.batch_size = 64
    cfg.train.grad_accum_steps = 2
    cfg.train.max_steps = 4000
    cfg.train.eval_interval = 100
    cfg.train.sample_interval = 100
    cfg.train.log_interval = 20
    cfg.train.eval_batches = 64
    cfg.train.weight_decay = 0.2

    cfg.sampling.max_new_tokens = 24
    cfg.sampling.temperature = 0.75
    cfg.sampling.top_p = 0.8

    cfg.runtime.device = "cuda"
    cfg.tracking.runs_dir = "runs"
    return cfg
