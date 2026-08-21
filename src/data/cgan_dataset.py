from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ShapeSample:
    """A generated shape image and its conditional label."""

    image: torch.Tensor
    label: int


class CGANShapeDataset(Dataset):
    """Synthetic dataset containing simple geometric shapes."""

    CLASS_NAMES = ("circle", "square", "triangle")

    def __init__(
        self,
        num_samples_per_class: int = 1000,
        image_size: int = 64,
        seed: int | None = 42,
    ) -> None:
        if num_samples_per_class <= 0:
            raise ValueError(
                "num_samples_per_class must be positive"
            )

        if image_size < 16:
            raise ValueError(
                "image_size must be at least 16"
            )

        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        self.images, self.labels = self._generate_dataset()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_NAMES)

    def _generate_dataset(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        labels = []

        for label in range(self.num_classes):
            for _ in range(self.num_samples_per_class):
                image = self._generate_shape(label)

                images.append(image)
                labels.append(label)

        image_tensor = torch.from_numpy(
            np.stack(images)
        ).unsqueeze(1)

        label_tensor = torch.tensor(
            labels,
            dtype=torch.long,
        )

        return image_tensor, label_tensor

    def _generate_shape(self, label: int) -> np.ndarray:
        image = np.zeros(
            (self.image_size, self.image_size),
            dtype=np.float32,
        )

        center_x = self.image_size // 2
        center_y = self.image_size // 2

        size = self.image_size // 4

        if label == 0:
            self._draw_circle(
                image,
                center_x,
                center_y,
                size,
            )
        elif label == 1:
            self._draw_square(
                image,
                center_x,
                center_y,
                size,
            )
        elif label == 2:
            self._draw_triangle(
                image,
                center_x,
                center_y,
                size,
            )
        else:
            raise ValueError(
                f"Unknown shape label: {label}"
            )

        return image

    @staticmethod
    def _draw_circle(
        image: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> None:
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]

        mask = (
            (x - center_x) ** 2
            + (y - center_y) ** 2
            <= radius ** 2
        )

        image[mask] = 1.0

    @staticmethod
    def _draw_square(
        image: np.ndarray,
        center_x: int,
        center_y: int,
        half_size: int,
    ) -> None:
        y_start = center_y - half_size
        y_end = center_y + half_size

        x_start = center_x - half_size
        x_end = center_x + half_size

        image[y_start:y_end, x_start:x_end] = 1.0

    @staticmethod
    def _draw_triangle(
        image: np.ndarray,
        center_x: int,
        center_y: int,
        size: int,
    ) -> None:
        top = center_y - size
        bottom = center_y + size

        for y in range(top, bottom + 1):
            if y < 0 or y >= image.shape[0]:
                continue

            progress = (y - top) / max(bottom - top, 1)

            half_width = int(progress * size)

            x_start = max(center_x - half_width, 0)
            x_end = min(
                center_x + half_width + 1,
                image.shape[1],
            )

            image[y, x_start:x_end] = 1.0