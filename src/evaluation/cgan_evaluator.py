from __future__ import annotations

import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.models.cgan_generator import CGANGenerator


class CGANEvaluator:
    """Evaluates generated CGAN samples against real dataset statistics."""

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

        labels = labels.to(self.device).long()

        batch_size = labels.shape[0]

        noise = torch.randn(
            batch_size,
            self.generator.noise_dim,
            device=self.device,
        )

        was_training = self.generator.training

        self.generator.eval()

        with torch.no_grad():
            generated_images = self.generator(
                noise,
                labels,
            )

        if was_training:
            self.generator.train()

        return generated_images.cpu()

    def _get_real_samples(
        self,
        dataset: CGANShapeDataset,
        class_index: int,
        samples_per_class: int,
    ) -> torch.Tensor:
        """Collect the requested number of real samples for one class."""

        real_samples = []

        for image, label in dataset:
            if int(label) == class_index:
                real_samples.append(image)

            if len(real_samples) == samples_per_class:
                break

        if len(real_samples) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class index {class_index}. "
                f"Requested {samples_per_class}, "
                f"found {len(real_samples)}."
            )

        return torch.stack(real_samples)

    def evaluate_generator(
        self,
        dataset: CGANShapeDataset,
        samples_per_class: int = 20,
    ) -> dict[str, dict[str, float | int]]:
        """
        Compare generated samples against real dataset statistics
        for each conditional class.
        """

        results = {}

        for class_index, class_name in enumerate(
            dataset.CLASS_NAMES
        ):
            real_images = self._get_real_samples(
                dataset=dataset,
                class_index=class_index,
                samples_per_class=samples_per_class,
            )

            labels = torch.full(
                (samples_per_class,),
                class_index,
                dtype=torch.long,
            )

            generated_images = self.generate_samples(
                labels=labels,
            )

            real_mean = real_images.mean().item()
            generated_mean = (
                generated_images.mean().item()
            )

            real_std = real_images.std().item()
            generated_std = (
                generated_images.std().item()
            )

            mean_difference = abs(
                real_mean - generated_mean
            )

            std_difference = abs(
                real_std - generated_std
            )

            results[class_name] = {
                "mean_pixel_intensity": generated_mean,
                "std_pixel_intensity": generated_std,
                "num_samples": samples_per_class,

                "real_mean_pixel_intensity": real_mean,
                "real_std_pixel_intensity": real_std,
                "num_real_samples": samples_per_class,

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
                    mean_difference
                ),
                "std_pixel_intensity_difference": (
                    std_difference
                ),
            }

        return results