from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam

from src.models.cgan_discriminator import CGANDiscriminator
from src.models.cgan_generator import CGANGenerator


class CGANTrainer:
    """Handles one-step adversarial training for a conditional GAN."""

    def __init__(
        self,
        generator: CGANGenerator,
        discriminator: CGANDiscriminator,
        device: str = "cpu",
        noise_dim: int = 100,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device(device)
        self.noise_dim = noise_dim

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.criterion = nn.BCELoss()

        self.generator_optimizer = Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
        )

        self.discriminator_optimizer = Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
        )

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """Run one discriminator update and one generator update."""

        self.generator.train()
        self.discriminator.train()

        images = images.to(self.device)
        labels = labels.to(self.device).long()

        batch_size = images.shape[0]

        real_targets = torch.ones(
            batch_size,
            1,
            device=self.device,
        )

        fake_targets = torch.zeros(
            batch_size,
            1,
            device=self.device,
        )

        # Train discriminator.
        self.discriminator_optimizer.zero_grad()

        real_predictions = self.discriminator(
            images,
            labels,
        )

        real_loss = self.criterion(
            real_predictions,
            real_targets,
        )

        noise = torch.randn(
            batch_size,
            self.noise_dim,
            device=self.device,
        )

        fake_images = self.generator(
            noise,
            labels,
        )

        fake_predictions = self.discriminator(
            fake_images.detach(),
            labels,
        )

        fake_loss = self.criterion(
            fake_predictions,
            fake_targets,
        )

        discriminator_loss = real_loss + fake_loss

        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Train generator.
        self.generator_optimizer.zero_grad()

        noise = torch.randn(
            batch_size,
            self.noise_dim,
            device=self.device,
        )

        generated_images = self.generator(
            noise,
            labels,
        )

        generator_predictions = self.discriminator(
            generated_images,
            labels,
        )

        generator_loss = self.criterion(
            generator_predictions,
            real_targets,
        )

        generator_loss.backward()
        self.generator_optimizer.step()

        return {
            "generator_loss": generator_loss.item(),
            "discriminator_loss": discriminator_loss.item(),
        }