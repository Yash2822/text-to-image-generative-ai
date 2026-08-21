import pytest
import torch

from src.models.cgan_generator import CGANGenerator


def test_generator_output_shape() -> None:
    generator = CGANGenerator()

    noise = torch.randn(4, 100)
    labels = torch.tensor([0, 1, 2, 0])

    images = generator(noise, labels)

    assert images.shape == (4, 1, 64, 64)


def test_generator_output_range() -> None:
    generator = CGANGenerator()

    noise = torch.randn(4, 100)
    labels = torch.tensor([0, 1, 2, 0])

    images = generator(noise, labels)

    assert torch.all(images >= 0)
    assert torch.all(images <= 1)


def test_generator_supports_different_batch_sizes() -> None:
    generator = CGANGenerator()
    generator.eval()

    for batch_size in [1, 2, 8]:
        noise = torch.randn(batch_size, 100)
        labels = torch.randint(0, 3, (batch_size,))

        with torch.no_grad():
            images = generator(noise, labels)

        assert images.shape == (batch_size, 1, 64, 64)

def test_generator_rejects_invalid_label() -> None:
    generator = CGANGenerator()

    noise = torch.randn(2, 100)
    labels = torch.tensor([0, 3])

    with pytest.raises(IndexError):
        generator(noise, labels)