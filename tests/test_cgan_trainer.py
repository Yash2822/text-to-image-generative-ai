import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.models.cgan_discriminator import CGANDiscriminator
from src.models.cgan_generator import CGANGenerator
from src.training.cgan_trainer import CGANTrainer


def test_trainer_initializes() -> None:
    generator, discriminator = CGANGenerator(), CGANDiscriminator()
    trainer = CGANTrainer(generator, discriminator, device="cpu")
    assert trainer.generator is generator
    assert trainer.discriminator is discriminator
    assert str(trainer.device) == "cpu"


def test_train_step_returns_finite_losses_and_updates_both_models() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=2)
    generator, discriminator = CGANGenerator(), CGANDiscriminator()
    trainer = CGANTrainer(generator, discriminator, device="cpu")
    images = torch.stack([dataset[i][0] for i in range(2)])
    labels = torch.tensor([dataset[i][1] for i in range(2)])
    g_before = next(generator.parameters()).detach().clone()
    d_before = next(discriminator.parameters()).detach().clone()
    losses = trainer.train_step(images, labels)
    assert set(losses) == {"generator_loss", "discriminator_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(g_before, next(generator.parameters()).detach())
    assert not torch.equal(d_before, next(discriminator.parameters()).detach())
    assert all(parameter.requires_grad for parameter in discriminator.parameters())


def test_train_step_supports_batch_size_one() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=1)
    trainer = CGANTrainer(CGANGenerator(), CGANDiscriminator(), device="cpu")
    image, label = dataset[0]
    losses = trainer.train_step(image.unsqueeze(0), torch.tensor([label]))
    assert all(value >= 0.0 for value in losses.values())
