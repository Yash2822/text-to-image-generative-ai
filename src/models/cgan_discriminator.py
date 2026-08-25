from __future__ import annotations

import torch
from torch import nn


class CGANDiscriminator(nn.Module):
    """Conditional discriminator for 64x64 grayscale shape images."""

    def __init__(
        self,
        image_size: int = 64,
        num_channels: int = 1,
        num_classes: int = 3,
        label_embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("image_size must be positive")

        if num_channels <= 0:
            raise ValueError("num_channels must be positive")

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if label_embedding_dim <= 0:
            raise ValueError("label_embedding_dim must be positive")

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.label_embedding_dim = label_embedding_dim

        self.image_dim = (
            num_channels * image_size * image_size
        )

        self.label_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=label_embedding_dim,
        )

        self.model = nn.Sequential(
            nn.Linear(
                self.image_dim + label_embedding_dim,
                1024,
            ),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return the probability that images are real."""

        expected_shape = (
            self.num_channels,
            self.image_size,
            self.image_size,
        )

        if images.ndim != 4:
            raise ValueError(
                "images must have shape "
                "[batch_size, channels, height, width]"
            )

        if tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected image shape {expected_shape}, "
                f"got {tuple(images.shape[1:])}"
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [batch_size]"
            )

        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                "images and labels must have the same batch size"
            )

        if labels.numel() > 0:
            if labels.min().item() < 0:
                raise ValueError(
                    "labels must be non-negative"
                )

            if labels.max().item() >= self.num_classes:
                raise ValueError(
                    "labels contain an invalid class index"
                )

        flattened_images = images.flatten(start_dim=1)

        label_embeddings = self.label_embedding(
            labels.long()
        )

        discriminator_input = torch.cat(
            [flattened_images, label_embeddings],
            dim=1,
        )

        return self.model(discriminator_input)