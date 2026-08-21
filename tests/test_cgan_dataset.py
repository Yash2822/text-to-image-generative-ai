import pytest
import torch

from src.data.cgan_dataset import CGANShapeDataset


def test_dataset_length() -> None:
    dataset = CGANShapeDataset(
        num_samples_per_class=10,
    )

    assert len(dataset) == 30


def test_number_of_classes() -> None:
    dataset = CGANShapeDataset()

    assert dataset.num_classes == 3
    assert dataset.CLASS_NAMES == (
        "circle",
        "square",
        "triangle",
    )


def test_sample_image_shape() -> None:
    dataset = CGANShapeDataset(
        num_samples_per_class=1,
        image_size=64,
    )

    image, label = dataset[0]

    assert image.shape == (1, 64, 64)
    assert image.dtype == torch.float32
    assert label == 0


def test_labels_are_valid() -> None:
    dataset = CGANShapeDataset(
        num_samples_per_class=5,
    )

    labels = [
        dataset[index][1]
        for index in range(len(dataset))
    ]

    assert set(labels) == {0, 1, 2}


def test_invalid_num_samples_raises_error() -> None:
    with pytest.raises(ValueError):
        CGANShapeDataset(
            num_samples_per_class=0,
        )


def test_small_image_size_raises_error() -> None:
    with pytest.raises(ValueError):
        CGANShapeDataset(
            image_size=8,
        )


def test_same_seed_is_reproducible() -> None:
    dataset_a = CGANShapeDataset(
        num_samples_per_class=2,
        seed=42,
    )

    dataset_b = CGANShapeDataset(
        num_samples_per_class=2,
        seed=42,
    )

    assert torch.equal(
        dataset_a.images,
        dataset_b.images,
    )

    assert torch.equal(
        dataset_a.labels,
        dataset_b.labels,
    )