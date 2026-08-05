"""Config loading: YAML files -> typed dataclasses.

Design:
    Every experiment is defined by a YAML file in configs/. We parse it into
    nested dataclasses instead of leaving it as a raw dict, so that:
      - typos in config keys raise an error immediately instead of silently
        returning None deep inside a training loop
      - editors/IDEs can autocomplete config.model.hidden_dim
      - defaults are explicit and documented in one place (this file)

    As new milestones add new components (GAN configs, diffusion fine-tuning
    configs, etc.), extend these dataclasses rather than passing raw dicts
    around. Keep this file as the single source of truth for what a valid
    config looks like.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class DataConfig:
    dataset_name: str = "oxford-102-flowers"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 4


@dataclasses.dataclass
class ModelConfig:
    name: str = "cgan_baseline"
    latent_dim: int = 100
    embedding_dim: int = 256
    num_classes: int = 102


@dataclasses.dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    checkpoint_dir: str = "experiments"
    log_every_n_steps: int = 50


@dataclasses.dataclass
class TrackingConfig:
    enabled: bool = False  # flip on once we start real training (Task 3+)
    project_name: str = "text-to-image-internship"
    run_name: str | None = None


@dataclasses.dataclass
class InferenceConfig:
    """Config for the pretrained Stable Diffusion baseline (provided project).

    Kept separate from ModelConfig: ModelConfig describes models we train
    ourselves (the CGAN); this describes a pretrained model we only run
    inference on (and later fine-tune in Task 5).
    """

    pretrained_model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    default_scheduler: str = "euler_a"
    output_dir: str = "outputs"


@dataclasses.dataclass
class ExperimentConfig:
    """Top-level config: composes the sections above.

    This is what code actually receives — never pass raw dicts into
    training/model code once this exists.
    """

    experiment_name: str = "unnamed_experiment"
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    tracking: TrackingConfig = dataclasses.field(default_factory=TrackingConfig)
    inference: InferenceConfig = dataclasses.field(default_factory=InferenceConfig)


def _build_section(section_cls: type, raw: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass section from a dict, ignoring absent keys."""
    if raw is None:
        return section_cls()
    valid_fields = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys for {section_cls.__name__}: {sorted(unknown)}"
        )
    return section_cls(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file into a typed ExperimentConfig.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        ValueError: if the YAML contains keys that don't map to any
            dataclass field (fail fast instead of silently ignoring typos).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "unnamed_experiment"),
        data=_build_section(DataConfig, raw.get("data")),
        model=_build_section(ModelConfig, raw.get("model")),
        training=_build_section(TrainingConfig, raw.get("training")),
        tracking=_build_section(TrackingConfig, raw.get("tracking")),
        inference=_build_section(InferenceConfig, raw.get("inference")),
    )
