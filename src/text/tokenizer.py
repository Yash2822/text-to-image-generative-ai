from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import CLIPTokenizer

from src.text.preprocessing import TextPreprocessor


@dataclass(frozen=True)
class TokenizedText:
    """Tokenized representation of a batch of text."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class CLIPTextTokenizer:
    """Project-level wrapper around the Hugging Face CLIP tokenizer."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.max_length = (
            self.tokenizer.model_max_length
            if max_length is None
            else max_length
        )

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def tokenize(self, texts: list[str]) -> TokenizedText:
        """Preprocess and tokenize a batch of texts."""

        processed = TextPreprocessor.preprocess_batch(texts)

        encoded = self.tokenizer(
            processed,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return TokenizedText(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )