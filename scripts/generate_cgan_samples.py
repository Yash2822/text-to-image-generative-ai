from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.models.cgan_generator import CGANGenerator


def main() -> None:
    seed = 42

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path("checkpoints/cgan_shapes.pt")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

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
    generator.eval()

    class_names = checkpoint["class_names"]

    samples_per_class = 10

    labels = torch.arange(
        checkpoint["num_classes"]
    ).repeat_interleave(samples_per_class)

    labels = labels.to(device)

    noise = torch.randn(
        len(labels),
        checkpoint["noise_dim"],
        device=device,
    )

    with torch.no_grad():
        generated_images = generator(
            noise,
            labels,
        )

    generated_images = generated_images.cpu()

    num_classes = checkpoint["num_classes"]

    figure, axes = plt.subplots(
        num_classes,
        samples_per_class,
        figsize=(15, 6),
    )

    for class_index in range(num_classes):
        for sample_index in range(samples_per_class):
            image_index = (
                class_index * samples_per_class
                + sample_index
            )

            axis = axes[
                class_index,
                sample_index,
            ]

            axis.imshow(
                generated_images[
                    image_index,
                    0,
                ],
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axis.axis("off")

            if sample_index == 0:
                axis.set_ylabel(
                    class_names[class_index],
                    rotation=0,
                    labelpad=40,
                    va="center",
                )

    figure.suptitle(
        "Conditional GAN Generated Shapes",
        fontsize=16,
    )

    figure.tight_layout()

    output_path = (
        output_dir / "cgan_generated_samples.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("=== CGAN SAMPLE GENERATION ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Random seed: {seed}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Generated samples: {len(labels)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()