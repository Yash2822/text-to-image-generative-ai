"""Real-model Stable Diffusion baseline smoke test.

Unlike the unit tests, this script downloads/loads the configured pretrained
model and performs one real inference run. Run it manually on a machine with
sufficient RAM/VRAM and Hugging Face access. It is intentionally excluded from
the normal pytest suite because it is slow and network/hardware dependent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.inference.diffusion_pipeline import StableDiffusionGenerator
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sd_baseline.yaml")
    parser.add_argument("--prompt", default="a red vintage sports car parked on a mountain road, cinematic photograph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generator = StableDiffusionGenerator(cfg.inference)

    if generator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    image, metadata = generator.generate_image(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        scheduler=cfg.inference.default_scheduler,
    )
    metadata["model_id"] = cfg.inference.pretrained_model_id
    metadata["experiment_name"] = cfg.experiment_name
    metadata["memory"] = generator.get_memory_usage()

    output_dir = Path(cfg.inference.output_dir) / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = generator.save_image(image, metadata, str(output_dir))
    json_path = Path(image_path).with_suffix("").with_name(Path(image_path).stem + "_metadata.json")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Baseline image: {image_path}")
    print(f"Structured metadata: {json_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
