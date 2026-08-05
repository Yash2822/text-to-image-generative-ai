"""Tests for src.inference.diffusion_pipeline.

These tests do NOT download real model weights (no network access to
Hugging Face is assumed). StableDiffusionPipeline.from_pretrained is
monkeypatched to return a lightweight fake pipe, so we can verify our
refactor preserves the original behavior: device selection, scheduler
switching, generation parameter handling, error handling, and saving —
without needing a GPU or an internet connection.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from src.inference import diffusion_pipeline as dp
from src.utils.config import InferenceConfig


class FakeScheduler:
    """Stand-in for a diffusers scheduler class."""

    def __init__(self, config=None):
        self.config = config or {}

    @classmethod
    def from_config(cls, config):
        return cls(config)

class FakeVAE:
    def __init__(self):
        self.slicing_enabled = False

    def enable_slicing(self):
        self.slicing_enabled = True

class FakePipe:
    """Stand-in for a diffusers StableDiffusionPipeline."""

    def __init__(self):
        self.scheduler = FakeScheduler()
        self.to_calls = []
        self.vae = FakeVAE()

    def enable_attention_slicing(self):
        pass

    def enable_vae_slicing(self):
        pass

    def enable_xformers_memory_efficient_attention(self):
        raise RuntimeError("xformers not available in test env")

    def to(self, device):
        self.to_calls.append(device)
        return self

    def enable_model_cpu_offload(self):
        pass

    def enable_sequential_cpu_offload(self):
        pass

    def __call__(self, **kwargs):
        img = Image.new("RGB", (kwargs["width"], kwargs["height"]))
        return SimpleNamespace(images=[img])


@pytest.fixture(autouse=True)
def patch_diffusers(monkeypatch):
    """Replace the real pipeline loader and scheduler classes with fakes."""
    monkeypatch.setattr(
        dp.StableDiffusionPipeline,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: FakePipe()),
    )
    fake_scheduler_map = {name: FakeScheduler for name in dp.SCHEDULER_MAP}
    monkeypatch.setattr(dp, "SCHEDULER_MAP", fake_scheduler_map)


# ---- device selection ----

def test_resolve_device_cpu_explicit():
    assert dp.resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_without_gpu(monkeypatch):
    monkeypatch.setattr(dp.torch.cuda, "is_available", lambda: False)
    assert dp.resolve_device("auto").type == "cpu"


# ---- initialization ----

def test_generator_initializes_with_cpu_config():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    assert gen.device.type == "cpu"
    assert gen.dtype == dp.torch.float32


def test_generator_applies_default_scheduler_on_init():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu", default_scheduler="ddim"))
    assert gen.current_scheduler == "ddim"


# ---- scheduler switching ----

def test_set_scheduler_valid_name_switches():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    result = gen.set_scheduler("dpm_solver")
    assert result is True
    assert gen.current_scheduler == "dpm_solver"


def test_set_scheduler_unknown_name_returns_false_and_keeps_current():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    previous = gen.current_scheduler
    result = gen.set_scheduler("not_a_real_scheduler")
    assert result is False
    assert gen.current_scheduler == previous


# ---- generation parameters & error handling ----

def test_generate_image_empty_prompt_raises_value_error():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    with pytest.raises(ValueError):
        gen.generate_image(prompt="   ")


def test_generate_image_rounds_dimensions_to_multiple_of_8():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    image, metadata = gen.generate_image(prompt="a red apple", width=513, height=511, num_inference_steps=1)
    assert metadata["width"] % 8 == 0
    assert metadata["height"] % 8 == 0
    assert image.size == (metadata["width"], metadata["height"])


def test_generate_image_returns_metadata_with_seed_and_scheduler():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    _, metadata = gen.generate_image(prompt="a blue car", seed=123, scheduler="ddim", num_inference_steps=1)
    assert metadata["seed"] == 123
    assert metadata["scheduler"] == "ddim"
    assert metadata["prompt"] == "a blue car"


def test_generate_image_random_seed_when_none():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    _, metadata = gen.generate_image(prompt="a green tree", seed=None, num_inference_steps=1)
    assert isinstance(metadata["seed"], int)


# ---- saving ----

def test_save_image_writes_png_and_metadata_file(tmp_path):
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu", output_dir=str(tmp_path)))
    image, metadata = gen.generate_image(prompt="a small house", num_inference_steps=1)
    filepath = gen.save_image(image, metadata)

    assert filepath.startswith(str(tmp_path))
    assert filepath.endswith(".png")
    saved_files = list(tmp_path.iterdir())
    assert any(f.suffix == ".png" for f in saved_files)
    assert any(f.name.endswith("_metadata.txt") for f in saved_files)


# ---- memory usage ----

def test_get_memory_usage_on_cpu_returns_note():
    gen = dp.StableDiffusionGenerator(InferenceConfig(device="cpu"))
    info = gen.get_memory_usage()
    assert info["device"] == "cpu"
