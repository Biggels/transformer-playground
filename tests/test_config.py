from transformer_playground.config import ExperimentConfig, apply_overrides


def test_apply_overrides_updates_nested_fields():
    cfg = ExperimentConfig()
    apply_overrides(
        cfg,
        [
            "model.n_layers=4",
            "model.mlp_enabled=false",
            "train.lr=0.001",
            "train.betas=0.8,0.9",
        ],
    )
    assert cfg.model.n_layers == 4
    assert cfg.model.mlp_enabled is False
    assert cfg.train.lr == 0.001
    assert cfg.train.betas == (0.8, 0.9)
