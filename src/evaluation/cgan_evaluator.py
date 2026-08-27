from __future__ import annotations

import math

import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.models.cgan_generator import CGANGenerator


class CGANEvaluator:
    """Evaluates a conditional GAN using pixel and shape-aware metrics."""

    def __init__(
        self,
        generator: CGANGenerator,
        device: str = "cpu",
    ) -> None:
        self.generator = generator
        self.device = torch.device(device)

        self.generator.to(self.device)

    def generate_samples(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generate images for the provided class labels."""

        self.generator.eval()

        labels = labels.to(self.device).long()

        noise = torch.randn(
            labels.size(0),
            self.generator.noise_dim,
            device=self.device,
        )

        with torch.no_grad():
            images = self.generator(
                noise,
                labels,
            )

        return images.cpu()

    @staticmethod
    def _get_shape_metrics(
        images: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Calculate shape-aware metrics for grayscale images."""

        foreground_areas = []
        bounding_box_aspect_ratios = []
        compactness_scores = []

        for image in images:
            mask = image[0] >= threshold

            foreground_area = (
                mask.float().mean().item()
            )

            foreground_areas.append(
                foreground_area
            )

            coordinates = torch.nonzero(
                mask,
                as_tuple=False,
            )

            if coordinates.numel() == 0:
                bounding_box_aspect_ratios.append(
                    0.0
                )

                compactness_scores.append(
                    0.0
                )

                continue

            min_y = coordinates[:, 0].min().item()
            max_y = coordinates[:, 0].max().item()

            min_x = coordinates[:, 1].min().item()
            max_x = coordinates[:, 1].max().item()

            height = max_y - min_y + 1
            width = max_x - min_x + 1

            aspect_ratio = (
                min(width, height)
                / max(width, height)
            )

            bounding_box_aspect_ratios.append(
                aspect_ratio
            )

            perimeter = 0

            for y, x in coordinates:
                y = y.item()
                x = x.item()

                neighbours = [
                    (y - 1, x),
                    (y + 1, x),
                    (y, x - 1),
                    (y, x + 1),
                ]

                for neighbour_y, neighbour_x in neighbours:
                    if (
                        neighbour_y < 0
                        or neighbour_y >= mask.shape[0]
                        or neighbour_x < 0
                        or neighbour_x >= mask.shape[1]
                        or not mask[
                            neighbour_y,
                            neighbour_x
                        ]
                    ):
                        perimeter += 1

            area = mask.sum().item()

            if perimeter > 0:
                compactness = (
                    4
                    * math.pi
                    * area
                    / (perimeter ** 2)
                )
            else:
                compactness = 0.0

            compactness_scores.append(
                compactness
            )

        return {
            "foreground_area_ratio": (
                sum(foreground_areas)
                / len(foreground_areas)
            ),
            "bounding_box_aspect_ratio": (
                sum(bounding_box_aspect_ratios)
                / len(bounding_box_aspect_ratios)
            ),
            "compactness": (
                sum(compactness_scores)
                / len(compactness_scores)
            ),
        }



    def evaluate_generator(
        self,
        dataset: CGANShapeDataset,
        samples_per_class: int = 20,
    ) -> dict[str, dict[str, float]]:
        """Compare real and generated samples for each class."""

        self.generator.eval()

        results = {}

        for class_index, class_name in enumerate(
            dataset.CLASS_NAMES
        ):
            real_images = []

            for image, label in dataset:
                if label == class_index:
                    real_images.append(image)

                    if (
                        len(real_images)
                        >= samples_per_class
                    ):
                        break

            real_images_tensor = torch.stack(
                real_images
            )

            labels = torch.full(
                (samples_per_class,),
                class_index,
                dtype=torch.long,
            )

            generated_images = self.generate_samples(
                labels
            )

            real_mean = (
                real_images_tensor.mean().item()
            )

            real_std = (
                real_images_tensor.std().item()
            )

            generated_mean = (
                generated_images.mean().item()
            )

            generated_std = (
                generated_images.std().item()
            )

            real_shape_metrics = (
                self._get_shape_metrics(
                    real_images_tensor
                )
            )

            generated_shape_metrics = (
                self._get_shape_metrics(
                    generated_images
                )
            )

            results[class_name] = {
                "real_mean_pixel_intensity": (
                    real_mean
                ),
                "real_std_pixel_intensity": (
                    real_std
                ),
                "num_real_samples": (
                    len(real_images)
                ),
                "generated_mean_pixel_intensity": (
                    generated_mean
                ),
                "generated_std_pixel_intensity": (
                    generated_std
                ),
                "num_generated_samples": (
                    samples_per_class
                ),
                "mean_pixel_intensity_difference": (
                    abs(
                        real_mean
                        - generated_mean
                    )
                ),
                "std_pixel_intensity_difference": (
                    abs(
                        real_std
                        - generated_std
                    )
                ),
                "real_foreground_area_ratio": (
                    real_shape_metrics[
                        "foreground_area_ratio"
                    ]
                ),
                "generated_foreground_area_ratio": (
                    generated_shape_metrics[
                        "foreground_area_ratio"
                    ]
                ),
                "foreground_area_ratio_difference": (
                    abs(
                        real_shape_metrics[
                            "foreground_area_ratio"
                        ]
                        - generated_shape_metrics[
                            "foreground_area_ratio"
                        ]
                    )
                ),
                "real_bounding_box_aspect_ratio": (
                    real_shape_metrics[
                        "bounding_box_aspect_ratio"
                    ]
                ),
                "generated_bounding_box_aspect_ratio": (
                    generated_shape_metrics[
                        "bounding_box_aspect_ratio"
                    ]
                ),
                "bounding_box_aspect_ratio_difference": (
                    abs(
                        real_shape_metrics[
                            "bounding_box_aspect_ratio"
                        ]
                        - generated_shape_metrics[
                            "bounding_box_aspect_ratio"
                        ]
                    )
                ),
                "real_compactness": (
                    real_shape_metrics[
                        "compactness"
                    ]
                ),
                "generated_compactness": (
                    generated_shape_metrics[
                        "compactness"
                    ]
                ),
                "compactness_difference": (
                    abs(
                        real_shape_metrics[
                            "compactness"
                        ]
                        - generated_shape_metrics[
                            "compactness"
                        ]
                    )
                ),
            }

        return results