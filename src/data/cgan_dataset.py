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
    """Seeded synthetic dataset with randomized geometric shapes."""

    CLASS_NAMES = ("circle", "square", "triangle")

    def __init__(
        self,
        num_samples_per_class: int = 1000,
        image_size: int = 64,
        seed: int | None = 42,
    ) -> None:
        if num_samples_per_class <= 0:
            raise ValueError("num_samples_per_class must be positive")
        if image_size < 16:
            raise ValueError("image_size must be at least 16")

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

    def _generate_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        images: list[np.ndarray] = []
        labels: list[int] = []
        for label in range(self.num_classes):
            for _ in range(self.num_samples_per_class):
                images.append(self._generate_shape(label))
                labels.append(label)
        return (
            torch.from_numpy(np.stack(images)).unsqueeze(1),
            torch.tensor(labels, dtype=torch.long),
        )

    def _generate_shape(self, label: int) -> np.ndarray:
        if label not in range(self.num_classes):
            raise ValueError(f"Unknown shape label: {label}")

        image = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32
        )
        size = int(self._rng.integers(
            max(3, self.image_size // 8),
            max(4, self.image_size // 3),
        ))
        margin = size + 1
        center_x = int(self._rng.integers(margin, self.image_size - margin))
        center_y = int(self._rng.integers(margin, self.image_size - margin))
        angle = float(self._rng.uniform(0.0, 2.0 * np.pi))

        if label == 0:
            self._draw_circle(image, center_x, center_y, size)
        elif label == 1:
            self._draw_square(image, center_x, center_y, size, angle)
        else:
            self._draw_triangle(image, center_x, center_y, size, angle)
        return image

    @staticmethod
    def _draw_circle(
        image: np.ndarray, center_x: int, center_y: int, radius: int
    ) -> None:
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        image[(x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2] = 1.0

    @staticmethod
    def _draw_square(
        image: np.ndarray,
        center_x: int,
        center_y: int,
        half_size: int,
        angle: float,
    ) -> None:
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        dx, dy = x - center_x, y - center_y
        local_x = dx * np.cos(angle) + dy * np.sin(angle)
        local_y = -dx * np.sin(angle) + dy * np.cos(angle)
        image[(np.abs(local_x) <= half_size) & (np.abs(local_y) <= half_size)] = 1.0

    @staticmethod
    def _draw_triangle(
        image: np.ndarray,
        center_x: int,
        center_y: int,
        size: int,
        angle: float,
    ) -> None:
        # Equilateral triangle vertices, rotated around its centroid.
        angles = angle + np.array([-np.pi / 2, np.pi / 6, 5 * np.pi / 6])
        vertices = np.stack(
            [
                center_x + size * np.cos(angles),
                center_y + size * np.sin(angles),
            ],
            axis=1,
        )
        y, x = np.mgrid[:image.shape[0], :image.shape[1]]
        px, py = x.astype(np.float32), y.astype(np.float32)
        ax, ay = vertices[0]
        bx, by = vertices[1]
        cx, cy = vertices[2]
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
        b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
        c = 1.0 - a - b
        image[(a >= 0) & (b >= 0) & (c >= 0)] = 1.0
