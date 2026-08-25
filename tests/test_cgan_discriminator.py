import pytest
import torch

from src.models.cgan_discriminator import CGANDiscriminator


def test_discriminator_output_shape() -> None:
    discriminator = CGANDiscriminator()

    images = torch.rand(4, 1, 64, 64)
    labels = torch.tensor([0, 1, 2, 0])

    output = discriminator(images, labels)

    assert output.shape == (4, 1)


def test_discriminator_output_range() -> None:
    discriminator = CGANDiscriminator()

    images = torch.rand(4, 1, 64, 64)
    labels = torch.tensor([0, 1, 2, 0])

    output = discriminator(images, labels)

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_discriminator_supports_different_batch_sizes(
    batch_size: int,
) -> None:
    discriminator = CGANDiscriminator()

    images = torch.rand(batch_size, 1, 64, 64)
    labels = torch.randint(0, 3, (batch_size,))

    output = discriminator(images, labels)

    assert output.shape == (batch_size, 1)


def test_discriminator_rejects_invalid_label_shape() -> None:
    discriminator = CGANDiscriminator()

    images = torch.rand(4, 1, 64, 64)
    labels = torch.tensor([[0, 1], [2, 0]])

    with pytest.raises(ValueError):
        discriminator(images, labels)


def test_discriminator_rejects_invalid_image_shape() -> None:
    discriminator = CGANDiscriminator()

    images = torch.rand(4, 64, 64)
    labels = torch.tensor([0, 1, 2, 0])

    with pytest.raises(ValueError):
        discriminator(images, labels)