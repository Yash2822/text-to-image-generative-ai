import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.models.cgan_discriminator import CGANDiscriminator
from src.models.cgan_generator import CGANGenerator
from src.training.cgan_trainer import CGANTrainer


def test_trainer_initializes() -> None:
    generator = CGANGenerator()
    discriminator = CGANDiscriminator()

    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device="cpu",
    )

    assert trainer.generator is generator
    assert trainer.discriminator is discriminator
    assert str(trainer.device) == "cpu"


def test_train_step_returns_losses() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=2)

    generator = CGANGenerator()
    discriminator = CGANDiscriminator()

    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device="cpu",
    )

    images = torch.stack(
        [dataset[index][0] for index in range(2)]
    )

    labels = torch.tensor(
        [dataset[index][1] for index in range(2)]
    )

    losses = trainer.train_step(images, labels)

    assert "generator_loss" in losses
    assert "discriminator_loss" in losses

    assert isinstance(losses["generator_loss"], float)
    assert isinstance(losses["discriminator_loss"], float)

    assert losses["generator_loss"] >= 0.0
    assert losses["discriminator_loss"] >= 0.0


def test_train_step_supports_batch_size_one() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=1)

    generator = CGANGenerator()
    discriminator = CGANDiscriminator()

    trainer = CGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device="cpu",
    )

    image, label = dataset[0]

    images = image.unsqueeze(0)
    labels = torch.tensor([label])

    losses = trainer.train_step(images, labels)

    assert losses["generator_loss"] >= 0.0
    assert losses["discriminator_loss"] >= 0.0