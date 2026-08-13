import torch
import pytest

from src.text.tokenizer import CLIPTextTokenizer, TokenizedText


@pytest.fixture(scope="module")
def tokenizer():
    return CLIPTextTokenizer()


def test_tokenizer_initializes(tokenizer):
    assert tokenizer.model_name == "openai/clip-vit-base-patch32"
    assert tokenizer.max_length == 77


def test_tokenize_returns_expected_type(tokenizer):
    result = tokenizer.tokenize(["A red car"])

    assert isinstance(result, TokenizedText)


def test_tokenize_returns_tensors(tokenizer):
    result = tokenizer.tokenize(
        [
            "A red car",
            "A black motorcycle",
        ]
    )

    assert isinstance(result.input_ids, torch.Tensor)
    assert isinstance(result.attention_mask, torch.Tensor)


def test_tokenize_returns_fixed_length(tokenizer):
    result = tokenizer.tokenize(
        [
            "A red car",
            "A black motorcycle",
        ]
    )

    assert result.input_ids.shape == (2, 77)
    assert result.attention_mask.shape == (2, 77)


def test_attention_mask_marks_padding(tokenizer):
    result = tokenizer.tokenize(["A red car"])

    mask = result.attention_mask[0]

    assert mask.shape == (77,)
    assert mask[0].item() == 1
    assert 0 in mask


def test_tokenizer_handles_multiple_texts(tokenizer):
    texts = [
        "A red car",
        "A black motorcycle",
        "A dog running through a park",
    ]

    result = tokenizer.tokenize(texts)

    assert result.input_ids.shape[0] == 3
    assert result.attention_mask.shape[0] == 3


def test_tokenizer_rejects_empty_text(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.tokenize([""])


def test_tokenizer_truncates_long_text(tokenizer):
    long_text = " ".join(["motorcycle"] * 200)

    result = tokenizer.tokenize([long_text])

    assert result.input_ids.shape == (1, 77)
    assert result.attention_mask.shape == (1, 77)