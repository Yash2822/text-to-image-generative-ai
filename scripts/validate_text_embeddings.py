from __future__ import annotations

from pathlib import Path

import torch

from src.data.coco_annotations import COCOAnnotationLoader
from src.text.clip_encoder import CLIPTextEncoder
from src.text.tokenizer import CLIPTextTokenizer


def main() -> None:
    coco_root = Path.home() / "fiftyone" / "coco-2017" / "raw"

    loader = COCOAnnotationLoader(
        instances_path=coco_root / "instances_val2017.json",
        captions_path=coco_root / "captions_val2017.json",
    )

    captions = []

    for image_id in list(loader.images_by_id)[:10]:
        image_captions = loader.get_captions(image_id)

        if image_captions:
            captions.append(image_captions[0])

        if len(captions) == 5:
            break
        
    print("=== TEXT EMBEDDING VALIDATION ===")
    print("Captions:", len(captions))

    for index, caption in enumerate(captions, start=1):
        print(f"{index}. {caption}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CLIPTextTokenizer()
    encoder = CLIPTextEncoder(device=device)

    tokenized = tokenizer.tokenize(captions)
    result = encoder.encode(tokenized)

    embeddings = result.embeddings

    print()
    print("Device:", embeddings.device)
    print("Dtype:", embeddings.dtype)
    print("Shape:", embeddings.shape)
    print("Mean:", embeddings.mean().item())
    print("Std:", embeddings.std().item())
    print("Min:", embeddings.min().item())
    print("Max:", embeddings.max().item())

    norms = torch.linalg.vector_norm(embeddings, dim=-1)

    print()
    print("Token embedding norms:")
    print("Mean:", norms.mean().item())
    print("Min:", norms.min().item())
    print("Max:", norms.max().item())


if __name__ == "__main__":
    main()