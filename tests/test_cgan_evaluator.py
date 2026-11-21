import pytest
import torch

from src.data.cgan_dataset import CGANShapeDataset
from src.evaluation.cgan_evaluator import CGANEvaluator
from src.models.cgan_generator import CGANGenerator


def test_evaluator_generates_expected_shape_and_range() -> None:
    evaluator = CGANEvaluator(CGANGenerator(), device="cpu")
    images = evaluator.generate_samples(torch.tensor([0, 1, 2]))
    assert images.shape == (3, 1, 64, 64)
    assert images.min() >= 0.0 and images.max() <= 1.0


def test_evaluator_returns_shape_conditioning_and_diversity_metrics() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=4, seed=7)
    evaluator = CGANEvaluator(CGANGenerator(), device="cpu")
    results = evaluator.evaluate_generator(dataset, samples_per_class=3)
    assert set(results) == {"circle", "square", "triangle"}
    for metrics in results.values():
        assert metrics["num_real_samples"] == 3
        assert metrics["num_generated_samples"] == 3
        assert 0.0 <= metrics["geometry_based_class_consistency"] <= 1.0
        assert metrics["generated_pixel_diversity"] >= 0.0
        assert "compactness_difference" in metrics


def test_evaluator_rejects_invalid_sample_count() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=2)
    evaluator = CGANEvaluator(CGANGenerator(), device="cpu")
    with pytest.raises(ValueError, match="positive"):
        evaluator.evaluate_generator(dataset, samples_per_class=0)
    with pytest.raises(ValueError, match="at least"):
        evaluator.evaluate_generator(dataset, samples_per_class=3)


def test_shape_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CGANEvaluator._get_shape_metrics(torch.empty(0, 1, 64, 64))
