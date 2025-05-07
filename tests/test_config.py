import pytest
from hydra import compose, initialize


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../protein_transformer/config", job_name="test_config"):
        cfg = compose(config_name="config")
        return cfg


def test_config_structure(cfg):
    assert "model" in cfg
    assert "lr" in cfg.model
    assert isinstance(cfg.model.lr, float)


def test_override_learning_rate():
    with initialize(config_path="../protein_transformer/config", job_name="test_override"):
        cfg = compose(config_name="config", overrides=["model.lr=5e-5"])
        assert cfg.model.lr == 5e-5
