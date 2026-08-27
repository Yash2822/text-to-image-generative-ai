from __future__ import annotations

from pathlib import Path

import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.evaluation.cgan_evaluator import CGANEvaluator
from src.models.cgan_generator import CGANGenerator


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path("checkpoints/cgan_shapes.pt")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    generator = CGANGenerator(
        latent_dim=checkpoint["noise_dim"],
        num_classes=checkpoint["num_classes"],
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.to(device)

    dataset = CGANShapeDataset(
        num_samples_per_class=100,
    )

    evaluator = CGANEvaluator(
        generator=generator,
        device=device,
    )

    samples_per_class = 20

    results = evaluator.evaluate_generator(
        dataset=dataset,
        samples_per_class=samples_per_class,
    )

    print("=== CGAN EVALUATION ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples per class: {samples_per_class}")
    print()

    for class_name, metrics in results.items():
        print(f"Class: {class_name}")
        print()

        print("Real samples")
        print(
            "Mean pixel intensity: "
            f"{metrics['real_mean_pixel_intensity']:.4f}"
        )
        print(
            "Std pixel intensity: "
            f"{metrics['real_std_pixel_intensity']:.4f}"
        )
        print(
            "Number of samples: "
            f"{metrics['num_real_samples']}"
        )

        print()
        print("Generated samples")
        print(
            "Mean pixel intensity: "
            f"{metrics['generated_mean_pixel_intensity']:.4f}"
        )
        print(
            "Std pixel intensity: "
            f"{metrics['generated_std_pixel_intensity']:.4f}"
        )
        print(
            "Number of samples: "
            f"{metrics['num_generated_samples']}"
        )

        print()
        print("Pixel metric differences")
        print(
            "Mean intensity difference: "
            f"{metrics['mean_pixel_intensity_difference']:.4f}"
        )
        print(
            "Std intensity difference: "
            f"{metrics['std_pixel_intensity_difference']:.4f}"
        )

        print()
        print("Shape-aware metrics")

        print(
            "Real foreground area ratio: "
            f"{metrics['real_foreground_area_ratio']:.4f}"
        )
        print(
            "Generated foreground area ratio: "
            f"{metrics['generated_foreground_area_ratio']:.4f}"
        )
        print(
            "Foreground area ratio difference: "
            f"{metrics['foreground_area_ratio_difference']:.4f}"
        )

        print(
            "Real bounding box aspect ratio: "
            f"{metrics['real_bounding_box_aspect_ratio']:.4f}"
        )
        print(
            "Generated bounding box aspect ratio: "
            f"{metrics['generated_bounding_box_aspect_ratio']:.4f}"
        )
        print(
            "Bounding box aspect ratio difference: "
            f"{metrics['bounding_box_aspect_ratio_difference']:.4f}"
        )

        print(
            "Real compactness: "
            f"{metrics['real_compactness']:.4f}"
        )
        print(
            "Generated compactness: "
            f"{metrics['generated_compactness']:.4f}"
    )
        print(
            "Compactness difference: "
            f"{metrics['compactness_difference']:.4f}"
        )

        print()
        print("-" * 40)
        print()


if __name__ == "__main__":
    main()