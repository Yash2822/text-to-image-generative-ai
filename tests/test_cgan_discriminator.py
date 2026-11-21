import pytest
import torch

from src.models.cgan_discriminator import CGANDiscriminator


def test_discriminator_output_shape() -> None:
    model = CGANDiscriminator()
    output = model(torch.rand(4, 1, 64, 64), torch.tensor([0, 1, 2, 0]))
    assert output.shape == (4, 1)


def test_discriminator_returns_logits_not_probabilities() -> None:
    model = CGANDiscriminator()
    output = model(torch.rand(8, 1, 64, 64), torch.randint(0, 3, (8,)))
    assert torch.any(output < 0.0) or torch.any(output > 1.0)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_discriminator_supports_batch_sizes(batch_size: int) -> None:
    model = CGANDiscriminator()
    output = model(
        torch.rand(batch_size, 1, 64, 64),
        torch.randint(0, 3, (batch_size,)),
    )
    assert output.shape == (batch_size, 1)


def test_discriminator_rejects_invalid_label_shape() -> None:
    model = CGANDiscriminator()
    with pytest.raises(ValueError):
        model(torch.rand(4, 1, 64, 64), torch.tensor([[0, 1], [2, 0]]))


def test_discriminator_rejects_invalid_image_shape() -> None:
    model = CGANDiscriminator()
    with pytest.raises(ValueError):
        model(torch.rand(4, 64, 64), torch.tensor([0, 1, 2, 0]))
