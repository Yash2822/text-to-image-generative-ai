"""Sanity tests for config loading.

Run with: pytest tests/

These aren't exhaustive ML tests (those come with models/training in later
milestones) — they exist to catch config regressions early, since every
other module depends on config loading working correctly.
"""

import pytest

from src.utils.config import ExperimentConfig, load_config


def test_base_config_loads():
    config = load_config("configs/base.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "base_example"
    assert config.data.image_size == 64
    assert config.model.num_classes == 102


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_unknown_key_raises():
    import tempfile
    from pathlib import Path

    bad_yaml = "experiment_name: test\ndata:\n  not_a_real_field: 123\n"
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = Path(tmp) / "bad.yaml"
        bad_path.write_text(bad_yaml)
        with pytest.raises(ValueError):
            load_config(bad_path)
