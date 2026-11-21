import pytest
import torch

from src.data.cgan_dataset import CGANShapeDataset


def test_dataset_length_and_classes() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=10)
    assert len(dataset) == 30
    assert dataset.num_classes == 3
    assert dataset.CLASS_NAMES == ("circle", "square", "triangle")


def test_sample_image_shape_and_range() -> None:
    image, label = CGANShapeDataset(num_samples_per_class=1)[0]
    assert image.shape == (1, 64, 64)
    assert image.dtype == torch.float32
    assert 0 <= label < 3
    assert image.min() >= 0 and image.max() <= 1


def test_invalid_dataset_parameters_raise() -> None:
    with pytest.raises(ValueError):
        CGANShapeDataset(num_samples_per_class=0)
    with pytest.raises(ValueError):
        CGANShapeDataset(image_size=8)


def test_same_seed_is_reproducible() -> None:
    a = CGANShapeDataset(num_samples_per_class=5, seed=42)
    b = CGANShapeDataset(num_samples_per_class=5, seed=42)
    assert torch.equal(a.images, b.images)
    assert torch.equal(a.labels, b.labels)


def test_different_seeds_produce_different_datasets() -> None:
    a = CGANShapeDataset(num_samples_per_class=5, seed=42)
    b = CGANShapeDataset(num_samples_per_class=5, seed=43)
    assert not torch.equal(a.images, b.images)


def test_samples_within_each_class_have_diversity() -> None:
    dataset = CGANShapeDataset(num_samples_per_class=12, seed=42)
    for class_index in range(dataset.num_classes):
        class_images = dataset.images[
            dataset.labels == class_index
        ]
        assert torch.unique(class_images.flatten(1), dim=0).shape[0] > 1
