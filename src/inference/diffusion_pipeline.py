"""Reusable Stable Diffusion inference wrapper.

This is the baseline provided by the internship, refactored out of the
original Colab notebook/script into an importable module. Behavior is
intentionally preserved as-is (device selection, memory optimizations,
scheduler switching, generation, saving) — this is a structural refactor,
not a redesign. It becomes the foundation Task 5 fine-tunes on top of, and
what Task 6's end-to-end pipeline calls into.

Changes from the original:
  - print() -> logging (see src/utils/logging.py)
  - hardcoded model_id/device -> driven by InferenceConfig (src/utils/config.py)
  - importable class instead of notebook-cell code
  - configured scheduler is explicitly applied during initialization, fixing a
    legacy bug where the pretrained pipeline scheduler could remain active
"""

from __future__ import annotations

import gc
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import torch
from torch import autocast
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
)

from src.utils.config import InferenceConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCHEDULER_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "ddim": DDIMScheduler,
    "dpm_solver": DPMSolverMultistepScheduler,
    "lms": LMSDiscreteScheduler,
}

SCHEDULER_INFO = {
    "euler_a": ("Euler Ancestral", "Fast, good for creative images"),
    "euler": ("Euler", "Deterministic, consistent results"),
    "ddim": ("DDIM", "Classic, good quality, slower"),
    "dpm_solver": ("DPM Solver", "High quality, efficient"),
    "lms": ("LMS", "Linear multistep, stable"),
}


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto'/'cuda'/'cpu' into a concrete torch.device.

    Pulled out as a standalone function (rather than a private method) so
    it can be unit-tested without loading any model weights.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("VRAM: %.1fGB", vram_gb)
        else:
            device = "cpu"
            logger.info("Using CPU (GPU not available)")
    return torch.device(device)


class StableDiffusionGenerator:
    """Loads a pretrained Stable Diffusion pipeline and generates images."""

    def __init__(self, config: InferenceConfig | None = None):
        self.config = config or InferenceConfig()
        try:
            self.device = resolve_device(self.config.device)
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

            logger.info("Initializing Stable Diffusion on %s", self.device)
            logger.info("Using precision: %s", self.dtype)

            self.pipe = self._load_pipeline(self.config.pretrained_model_id)
            self.current_scheduler = self.config.default_scheduler
            self.set_scheduler(self.current_scheduler)
            logger.info("Stable Diffusion Generator ready")
        except Exception as e:
            logger.error("Initialization error: %s", e)
            raise

    def _load_pipeline(self, model_id: str) -> StableDiffusionPipeline:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            logger.info("Applying memory optimizations")
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()

            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("XFormers attention: enabled")
            except Exception as e:
                logger.info("XFormers not available: %s", e)

            if self.device.type == "cuda":
                try:
                    pipe = pipe.to(self.device)
                    logger.info("Full GPU loading: success")
                except RuntimeError:
                    logger.warning("GPU memory limited: using CPU offload")
                    pipe.enable_model_cpu_offload()
            else:
                pipe.enable_sequential_cpu_offload()
                logger.info("CPU sequential offload: enabled")
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def set_scheduler(self, scheduler_name: str) -> bool:
        if scheduler_name not in SCHEDULER_MAP:
            logger.warning("Unknown scheduler: %s", scheduler_name)
            return False
        try:
            scheduler_class = SCHEDULER_MAP[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
            self.current_scheduler = scheduler_name
            name, desc = SCHEDULER_INFO[scheduler_name]
            logger.info("Scheduler changed: %s (%s)", name, desc)
            return True
        except Exception as e:
            logger.error("Scheduler error: %s", e)
            return False

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        scheduler: str = "euler_a",
    ) -> Tuple[Image.Image, dict]:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        self.set_scheduler(scheduler)
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info("Generating: '%s...'", prompt[:50])
        logger.info(
            "Size: %dx%d, Steps: %d, CFG: %.1f, Seed: %d, Scheduler: %s",
            width, height, num_inference_steps, guidance_scale, seed, scheduler,
        )

        start_time = time.time()
        try:
            with torch.inference_mode():
                gen_kwargs = dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                if self.device.type == "cuda" and self.dtype == torch.float16:
                    with autocast(self.device.type):
                        result = self.pipe(**gen_kwargs)
                else:
                    result = self.pipe(**gen_kwargs)

            generation_time = time.time() - start_time
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "device": str(self.device),
                "dtype": str(self.dtype),
            }
            logger.info("Generated in %.2fs", generation_time)
            return result.images[0], metadata

        except torch.cuda.OutOfMemoryError as e:
            self._cleanup_memory()
            raise RuntimeError(
                "GPU out of memory. Try reducing image size, fewer steps, "
                "or use CPU mode."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}") from e
        finally:
            self._cleanup_memory()

    def _cleanup_memory(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> dict:
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            }
        return {"device": "cpu", "note": "CPU memory tracking not available"}

    def save_image(self, image: Image.Image, metadata: dict, output_dir: str | None = None) -> str:
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sd_gen_{timestamp}_s{metadata['seed']}_{metadata['width']}x{metadata['height']}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)

        metadata_file = filepath.replace(".png", "_metadata.txt")
        with open(metadata_file, "w") as f:
            f.write("Stable Diffusion Generation Metadata\n")
            f.write("=" * 40 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        logger.info("Saved: %s", filepath)
        return filepath
