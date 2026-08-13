import torch

from src.text.clip_encoder import CLIPTextEncoder, TextEmbeddings
from src.text.tokenizer import CLIPTextTokenizer


def test_encoder_initializes():
    encoder = CLIPTextEncoder()

    assert encoder.model_name == "openai/clip-vit-base-patch32"
    assert encoder.hidden_size == 512
    assert encoder.max_length == 77


def test_encoder_is_in_evaluation_mode():
    encoder = CLIPTextEncoder()

    assert encoder.model.training is False


def test_encoder_parameters_are_frozen():
    encoder = CLIPTextEncoder()

    assert all(
        parameter.requires_grad is False
        for parameter in encoder.model.parameters()
    )


def test_encoder_returns_text_embeddings():
    tokenizer = CLIPTextTokenizer()
    encoder = CLIPTextEncoder()

    tokenized = tokenizer.tokenize(
        ["A red car", "A black motorcycle"]
    )

    result = encoder.encode(tokenized)

    assert isinstance(result, TextEmbeddings)
    assert isinstance(result.embeddings, torch.Tensor)


def test_encoder_returns_expected_shape():
    tokenizer = CLIPTextTokenizer()
    encoder = CLIPTextEncoder()

    tokenized = tokenizer.tokenize(
        ["A red car", "A black motorcycle"]
    )

    result = encoder.encode(tokenized)

    assert result.embeddings.shape == (2, 77, 512)


def test_encoder_handles_single_text():
    tokenizer = CLIPTextTokenizer()
    encoder = CLIPTextEncoder()

    tokenized = tokenizer.tokenize(["A red car"])

    result = encoder.encode(tokenized)

    assert result.embeddings.shape == (1, 77, 512)


def test_encoder_is_deterministic():
    tokenizer = CLIPTextTokenizer()
    encoder = CLIPTextEncoder()

    tokenized = tokenizer.tokenize(["A red car"])

    first = encoder.encode(tokenized)
    second = encoder.encode(tokenized)

    assert torch.equal(
        first.embeddings,
        second.embeddings,
    )


def test_encoder_uses_requested_device():
    tokenizer = CLIPTextTokenizer()
    encoder = CLIPTextEncoder(device="cpu")

    tokenized = tokenizer.tokenize(["A red car"])
    result = encoder.encode(tokenized)

    assert result.embeddings.device.type == "cpu"