from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.cgan_dataset import CGANShapeDataset
from src.models.cgan_discriminator import CGANDiscriminator
from src.models.cgan_generator import CGANGenerator
from src.training.cgan_trainer import CGANTrainer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CGANShapeDataset(
        num_samples_per_class=100,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    generator = CGANGenerator()
    discriminator = CGANDiscriminator()

    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
    )

    num_epochs = 50

    print("=== CGAN TRAINING ===")
    print(f"Device: {device}")
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {num_epochs}")
    print()

    for epoch in range(1, num_epochs + 1):
        generator_losses = []
        discriminator_losses = []

        for images, labels in dataloader:
            losses = trainer.train_step(
                images,
                labels,
            )

            generator_losses.append(
                losses["generator_loss"]
            )

            discriminator_losses.append(
                losses["discriminator_loss"]
            )

        average_generator_loss = sum(
            generator_losses
        ) / len(generator_losses)

        average_discriminator_loss = sum(
            discriminator_losses
        ) / len(discriminator_losses)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"G Loss: {average_generator_loss:.4f} | "
            f"D Loss: {average_discriminator_loss:.4f}"
        )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "cgan_shapes.pt"

    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "num_epochs": num_epochs,
            "noise_dim": trainer.noise_dim,
            "num_classes": dataset.num_classes,
            "class_names": dataset.CLASS_NAMES,
        },
        checkpoint_path,
    )

    print()
    print("Training complete.")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()