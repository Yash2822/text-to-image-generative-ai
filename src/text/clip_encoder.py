from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import CLIPTextModel

from src.text.tokenizer import TokenizedText


@dataclass(frozen=True)
class TextEmbeddings:
    """CLIP text encoder output."""

    embeddings: torch.Tensor


class CLIPTextEncoder:
    """Encodes tokenized text using a pretrained CLIP text encoder."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | torch.device = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)

        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def hidden_size(self) -> int:
        """Return the encoder hidden dimension."""
        return self.model.config.hidden_size

    @property
    def max_length(self) -> int:
        """Return the maximum supported sequence length."""
        return self.model.config.max_position_embeddings

    def encode(self, tokenized: TokenizedText) -> TextEmbeddings:
        """Encode tokenized text into token-level CLIP representations."""

        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return TextEmbeddings(
            embeddings=outputs.last_hidden_state,
        )