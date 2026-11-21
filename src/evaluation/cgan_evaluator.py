from __future__ import annotations

import math

import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.models.cgan_generator import CGANGenerator


class CGANEvaluator:
    """Evaluate pixel statistics, geometry, conditioning, and sample diversity."""

    def __init__(self, generator: CGANGenerator, device: str = "cpu") -> None:
        self.generator = generator
        self.device = torch.device(device)
        self.generator.to(self.device)

    def generate_samples(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1 or labels.numel() == 0:
            raise ValueError("labels must be a non-empty 1D tensor")
        self.generator.eval()
        labels = labels.to(self.device).long()
        noise = torch.randn(
            labels.size(0), self.generator.noise_dim, device=self.device
        )
        with torch.no_grad():
            images = self.generator(noise, labels)
        return images.cpu()

    @staticmethod
    def _get_shape_metrics(
        images: torch.Tensor, threshold: float = 0.5
    ) -> dict[str, float]:
        """Calculate mean foreground area, bbox ratio, and compactness."""
        if images.ndim != 4 or images.shape[1] != 1 or images.shape[0] == 0:
            raise ValueError("images must be non-empty [N, 1, H, W] tensor")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        areas: list[float] = []
        aspect_ratios: list[float] = []
        compactness_scores: list[float] = []
        for image in images:
            mask = image[0] >= threshold
            areas.append(float(mask.float().mean().item()))
            coords = torch.nonzero(mask, as_tuple=False)
            if coords.numel() == 0:
                aspect_ratios.append(0.0)
                compactness_scores.append(0.0)
                continue

            min_y, max_y = coords[:, 0].min().item(), coords[:, 0].max().item()
            min_x, max_x = coords[:, 1].min().item(), coords[:, 1].max().item()
            height, width = max_y - min_y + 1, max_x - min_x + 1
            aspect_ratios.append(float(min(width, height) / max(width, height)))

            # Four-neighbour perimeter, computed with tensor operations.
            padded = torch.nn.functional.pad(
                mask[None, None], (1, 1, 1, 1)
            )[0, 0]
            perimeter = (
                (mask & ~padded[:-2, 1:-1]).sum()
                + (mask & ~padded[2:, 1:-1]).sum()
                + (mask & ~padded[1:-1, :-2]).sum()
                + (mask & ~padded[1:-1, 2:]).sum()
            ).item()
            area = float(mask.sum().item())
            compactness_scores.append(
                float(4.0 * math.pi * area / (perimeter ** 2))
                if perimeter > 0 else 0.0
            )

        return {
            "foreground_area_ratio": sum(areas) / len(areas),
            "bounding_box_aspect_ratio": sum(aspect_ratios) / len(aspect_ratios),
            "compactness": sum(compactness_scores) / len(compactness_scores),
        }

    @staticmethod
    def _feature_vector(metrics: dict[str, float]) -> torch.Tensor:
        return torch.tensor([
            metrics["foreground_area_ratio"],
            metrics["bounding_box_aspect_ratio"],
            metrics["compactness"],
        ], dtype=torch.float32)

    def evaluate_generator(
        self,
        dataset: CGANShapeDataset,
        samples_per_class: int = 20,
    ) -> dict[str, dict[str, float]]:
        """Compare generated samples with real data for each requested class."""
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")
        if dataset.num_samples_per_class < samples_per_class:
            raise ValueError(
                "dataset must contain at least samples_per_class examples per class"
            )

        self.generator.eval()
        real_features: list[torch.Tensor] = []
        for class_index in range(dataset.num_classes):
            class_images = dataset.images[dataset.labels == class_index]
            real_features.append(
                self._feature_vector(self._get_shape_metrics(class_images))
            )
        centroids = torch.stack(real_features)
        feature_scale = centroids.std(dim=0, unbiased=False).clamp_min(1e-6)

        results: dict[str, dict[str, float]] = {}
        for class_index, class_name in enumerate(dataset.CLASS_NAMES):
            real_images = dataset.images[
                dataset.labels == class_index
            ][:samples_per_class]
            labels = torch.full(
                (samples_per_class,), class_index, dtype=torch.long
            )
            generated_images = self.generate_samples(labels)

            real_mean = float(real_images.mean().item())
            real_std = float(real_images.std(unbiased=False).item())
            generated_mean = float(generated_images.mean().item())
            generated_std = float(generated_images.std(unbiased=False).item())
            real_shape = self._get_shape_metrics(real_images)
            generated_shape = self._get_shape_metrics(generated_images)

            generated_features = torch.stack([
                self._feature_vector(self._get_shape_metrics(image.unsqueeze(0)))
                for image in generated_images
            ])
            distances = torch.cdist(
                generated_features / feature_scale,
                centroids / feature_scale,
            )
            predicted_classes = distances.argmin(dim=1)
            conditioning_accuracy = float(
                (predicted_classes == class_index).float().mean().item()
            )
            diversity = float(
                generated_images.std(dim=0, unbiased=False).mean().item()
            )

            metrics: dict[str, float] = {
                "real_mean_pixel_intensity": real_mean,
                "real_std_pixel_intensity": real_std,
                "num_real_samples": float(len(real_images)),
                "generated_mean_pixel_intensity": generated_mean,
                "generated_std_pixel_intensity": generated_std,
                "num_generated_samples": float(len(generated_images)),
                "mean_pixel_intensity_difference": abs(real_mean - generated_mean),
                "std_pixel_intensity_difference": abs(real_std - generated_std),
                "generated_pixel_diversity": diversity,
                "geometry_based_class_consistency": conditioning_accuracy,
            }
            for key, value in real_shape.items():
                metrics[f"real_{key}"] = value
            for key, value in generated_shape.items():
                metrics[f"generated_{key}"] = value
                metrics[f"{key}_difference"] = abs(real_shape[key] - value)
            results[class_name] = metrics
        return results
