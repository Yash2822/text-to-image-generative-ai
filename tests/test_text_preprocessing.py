import pytest

from src.text.preprocessing import TextPreprocessor


def test_preprocess_strips_outer_whitespace():
    result = TextPreprocessor.preprocess(
        "   A motorcycle outside   "
    )

    assert result == "A motorcycle outside"


def test_preprocess_rejects_empty_text():
    with pytest.raises(ValueError):
        TextPreprocessor.preprocess("")


def test_preprocess_rejects_whitespace_only_text():
    with pytest.raises(ValueError):
        TextPreprocessor.preprocess("   ")


def test_preprocess_rejects_non_string():
    with pytest.raises(TypeError):
        TextPreprocessor.preprocess(123)


def test_preprocess_batch():
    texts = [
        "  A motorcycle  ",
        "A red car",
    ]

    result = TextPreprocessor.preprocess_batch(texts)

    assert result == [
        "A motorcycle",
        "A red car",
    ]


def test_preprocess_batch_rejects_invalid_item():
    with pytest.raises(TypeError):
        TextPreprocessor.preprocess_batch(
            ["A motorcycle", 123]
        )