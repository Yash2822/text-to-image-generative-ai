from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam

from src.models.cgan_discriminator import CGANDiscriminator
from src.models.cgan_generator import CGANGenerator


class CGANTrainer:
    """One discriminator update followed by one generator update."""

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
        if noise_dim <= 0:
            raise ValueError("noise_dim must be positive")
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device(device)
        self.noise_dim = noise_dim
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.generator_optimizer = Adam(
            self.generator.parameters(), lr=learning_rate, betas=(beta1, beta2)
        )
        self.discriminator_optimizer = Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2)
        )

    def train_step(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        """Run one discriminator update and one generator update."""
        self.generator.train()
        self.discriminator.train()
        images = images.to(self.device)
        labels = labels.to(self.device).long()
        batch_size = images.shape[0]
        if batch_size == 0:
            raise ValueError("training batch must not be empty")

        real_targets = torch.ones(batch_size, 1, device=self.device)
        fake_targets = torch.zeros(batch_size, 1, device=self.device)

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        real_logits = self.discriminator(images, labels)
        real_loss = self.criterion(real_logits, real_targets)

        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(noise, labels)
        fake_logits = self.discriminator(fake_images, labels)
        fake_loss = self.criterion(fake_logits, fake_targets)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad(set_to_none=True)
        # Freeze D parameters, but keep autograd through D(input) so G receives gradients.
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            generated_images = self.generator(noise, labels)
            generated_logits = self.discriminator(generated_images, labels)
            generator_loss = self.criterion(generated_logits, real_targets)
            generator_loss.backward()
            self.generator_optimizer.step()
        finally:
            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(True)

        return {
            "generator_loss": float(generator_loss.detach().item()),
            "discriminator_loss": float(discriminator_loss.detach().item()),
        }
