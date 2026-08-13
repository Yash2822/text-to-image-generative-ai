from __future__ import annotations


class TextPreprocessor:
    """Validates and lightly normalizes text before tokenization."""

    @staticmethod
    def preprocess(text: str) -> str:
        """Validate and normalize a single text input."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        cleaned = text.strip()

        if not cleaned:
            raise ValueError("text cannot be empty")

        return cleaned

    @classmethod
    def preprocess_batch(cls, texts: list[str]) -> list[str]:
        """Validate and normalize a batch of text inputs."""

        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        return [cls.preprocess(text) for text in texts]