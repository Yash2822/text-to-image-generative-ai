from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cgan_dataset import CGANShapeDataset
from src.models.cgan_discriminator import CGANDiscriminator
from src.models.cgan_generator import CGANGenerator
from src.training.cgan_trainer import CGANTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed = 42
    num_epochs = 50
    batch_size = 32
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CGANShapeDataset(num_samples_per_class=100, seed=seed)
    loader_generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    generator = CGANGenerator()
    discriminator = CGANDiscriminator()
    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
    )

    history: list[dict[str, float | int]] = []
    print("=== CGAN TRAINING ===")
    print(f"Device: {device}")
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {num_epochs}")
    print(f"Seed: {seed}")
    print()

    for epoch in range(1, num_epochs + 1):
        generator_losses: list[float] = []
        discriminator_losses: list[float] = []
        for images, labels in dataloader:
            losses = trainer.train_step(images, labels)
            generator_losses.append(losses["generator_loss"])
            discriminator_losses.append(losses["discriminator_loss"])

        avg_g = float(np.mean(generator_losses))
        avg_d = float(np.mean(discriminator_losses))
        history.append({
            "epoch": epoch,
            "generator_loss": avg_g,
            "discriminator_loss": avg_d,
        })
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}"
        )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "cgan_shapes_phase3_batch1.pt"
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "generator_optimizer_state_dict": trainer.generator_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": trainer.discriminator_optimizer.state_dict(),
            "num_epochs": num_epochs,
            "noise_dim": trainer.noise_dim,
            "num_classes": dataset.num_classes,
            "class_names": dataset.CLASS_NAMES,
            "seed": seed,
            "image_size": dataset.image_size,
            "num_samples_per_class": dataset.num_samples_per_class,
            "history": history,
            "loss": "BCEWithLogitsLoss",
        },
        checkpoint_path,
    )
    print()
    print("Training complete.")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
