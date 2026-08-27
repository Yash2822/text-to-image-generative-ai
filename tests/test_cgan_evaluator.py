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
        assert "real_mean_pixel_intensity" in metrics
        assert "real_std_pixel_intensity" in metrics
        assert "num_real_samples" in metrics

        assert "generated_mean_pixel_intensity" in metrics
        assert "generated_std_pixel_intensity" in metrics
        assert "num_generated_samples" in metrics

        assert "mean_pixel_intensity_difference" in metrics
        assert "std_pixel_intensity_difference" in metrics

        assert metrics["num_real_samples"] == 2
        assert metrics["num_generated_samples"] == 2

        assert isinstance(
            metrics["real_mean_pixel_intensity"],
            float,
        )

        assert isinstance(
            metrics["real_std_pixel_intensity"],
            float,
        )

        assert isinstance(
            metrics["generated_mean_pixel_intensity"],
            float,
        )

        assert isinstance(
            metrics["generated_std_pixel_intensity"],
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
def test_shape_metrics_return_expected_keys() -> None:
    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    images = torch.zeros(
        2,
        1,
        64,
        64,
    )

    images[:, :, 20:40, 20:40] = 1.0

    metrics = evaluator._get_shape_metrics(
        images
    )

    assert "foreground_area_ratio" in metrics
    assert "bounding_box_aspect_ratio" in metrics
    assert "compactness" in metrics


def test_shape_metrics_are_valid() -> None:
    generator = CGANGenerator()

    evaluator = CGANEvaluator(
        generator=generator,
        device="cpu",
    )

    images = torch.zeros(
        2,
        1,
        64,
        64,
    )

    images[:, :, 20:40, 20:40] = 1.0

    metrics = evaluator._get_shape_metrics(
        images
    )

    assert (
        0.0
        <= metrics["foreground_area_ratio"]
        <= 1.0
    )

    assert (
        0.0
        <= metrics["bounding_box_aspect_ratio"]
        <= 1.0
    )

    assert metrics["compactness"] >= 0.0


def test_evaluation_returns_shape_aware_metrics() -> None:
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

    required_metrics = [
        "real_foreground_area_ratio",
        "generated_foreground_area_ratio",
        "foreground_area_ratio_difference",
        "real_bounding_box_aspect_ratio",
        "generated_bounding_box_aspect_ratio",
        "bounding_box_aspect_ratio_difference",
        "real_compactness",
        "generated_compactness",
        "compactness_difference",
    ]

    for metrics in results.values():
        for metric_name in required_metrics:
            assert metric_name in metrics


def test_shape_metric_differences_are_non_negative() -> None:
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

    for metrics in results.values():
        assert (
            metrics[
                "foreground_area_ratio_difference"
            ]
            >= 0.0
        )

        assert (
            metrics[
                "bounding_box_aspect_ratio_difference"
            ]
            >= 0.0
        )

        assert (
            metrics["compactness_difference"]
            >= 0.0
        )