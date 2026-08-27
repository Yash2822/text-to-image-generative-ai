from __future__ import annotations

import torch
from torch import nn


class CGANGenerator(nn.Module):
    """Conditional generator for 64x64 grayscale shape images."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 3,
        embedding_dim: int = 10,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.noise_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.label_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=embedding_dim,
        )

        self.project = nn.Sequential(
            nn.Linear(
                latent_dim + embedding_dim,
                256 * 4 * 4,
            ),
            nn.LayerNorm(256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generate 64x64 grayscale images conditioned on class labels."""

        label_embeddings = self.label_embedding(
            labels.long()
        )

        generator_input = torch.cat(
            [noise, label_embeddings],
            dim=1,
        )

        x = self.project(generator_input)

        x = x.view(
            noise.size(0),
            256,
            4,
            4,
        )

        return self.generator(x)