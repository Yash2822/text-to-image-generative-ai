import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.evaluation.cgan_evaluator import CGANEvaluator
from src.models.cgan_generator import CGANGenerator


def test_evaluator_initializes() -> None:
    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    assert evaluator.generator is generator
    assert str(evaluator.device) == "cpu"


def test_generate_samples_returns_correct_shape() -> None:
    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    labels = torch.tensor([0, 1, 2])

    images = evaluator.generate_samples(
        labels=labels,
    )

    assert images.shape == (3, 1, 64, 64)


def test_generate_samples_returns_valid_range() -> None:
    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    labels = torch.tensor([0, 1, 2])

    images = evaluator.generate_samples(
        labels=labels,
    )

    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0


def test_evaluate_generator_returns_class_metrics() -> None:
    dataset = CGANShapeDataset(
        num_samples_per_class=2,
    )

    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    results = evaluator.evaluate_generator(
        dataset=dataset,
        samples_per_class=2,
    )

    assert "circle" in results
    assert "square" in results
    assert "triangle" in results

    for class_name, metrics in results.items():
        assert "mean_pixel_intensity" in metrics
        assert "std_pixel_intensity" in metrics
        assert "num_samples" in metrics

        assert metrics["num_samples"] == 2

        assert isinstance(
            metrics["mean_pixel_intensity"],
            float,
        )

        assert isinstance(
            metrics["std_pixel_intensity"],
            float,
        )


def test_evaluate_generator_returns_real_vs_generated_metrics() -> None:
    dataset = CGANShapeDataset(
        num_samples_per_class=3,
    )

    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    results = evaluator.evaluate_generator(
        dataset=dataset,
        samples_per_class=2,
    )

    for class_name, metrics in results.items():
        assert "real_mean_pixel_intensity" in metrics
        assert "generated_mean_pixel_intensity" in metrics
        assert "mean_pixel_intensity_difference" in metrics

        assert "real_std_pixel_intensity" in metrics
        assert "generated_std_pixel_intensity" in metrics
        assert "std_pixel_intensity_difference" in metrics

        assert "num_real_samples" in metrics
        assert "num_generated_samples" in metrics


def test_evaluate_generator_uses_requested_sample_count() -> None:
    dataset = CGANShapeDataset(
        num_samples_per_class=5,
    )

    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    results = evaluator.evaluate_generator(
        dataset=dataset,
        samples_per_class=3,
    )

    for class_name, metrics in results.items():
        assert metrics["num_real_samples"] == 3
        assert metrics["num_generated_samples"] == 3


def test_metric_differences_are_non_negative() -> None:
    dataset = CGANShapeDataset(
        num_samples_per_class=2,
    )

    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    results = evaluator.evaluate_generator(
        dataset=dataset,
        samples_per_class=2,
    )

    for class_name, metrics in results.items():
        assert (
            metrics["mean_pixel_intensity_difference"]
            >= 0.0
        )

        assert (
            metrics["std_pixel_intensity_difference"]
            >= 0.0
        )