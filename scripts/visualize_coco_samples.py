from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.data.coco_annotations import COCOAnnotationLoader
from src.data.coco_sampling import COCOSampler


def main():
    coco_root = Path.home() / "fiftyone" / "coco-2017"
    raw_dir = coco_root / "raw"
    image_dir = coco_root / "validation" / "data"

    output_dir = Path("outputs") / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = COCOAnnotationLoader(
        instances_path=raw_dir / "instances_val2017.json",
        captions_path=raw_dir / "captions_val2017.json",
    )

    sampler = COCOSampler(loader)
    samples = sampler.representative_samples()

    fig, axes = plt.subplots(
        len(samples),
        1,
        figsize=(12, 18),
    )

    for ax, sample in zip(axes, samples):
        image_info = loader.images_by_id[sample.image_id]
        image_path = image_dir / image_info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        annotations = loader.get_annotations(sample.image_id)
        captions = loader.get_captions(sample.image_id)

        categories = [
            loader.get_category_name(annotation["category_id"])
            for annotation in annotations
        ]

        unique_categories = sorted(set(categories))

        ax.imshow(image)
        ax.axis("off")

        title = (
            f"{sample.reason}\n"
            f"ID: {sample.image_id} | "
            f"Size: {image_info['width']}x{image_info['height']} | "
            f"Objects: {len(annotations)}"
        )

        ax.set_title(title, fontsize=11)

        caption = captions[0] if captions else "No caption"

        category_text = (
            ", ".join(unique_categories)
            if unique_categories
            else "None"
        )

        ax.text(
            0.5,
            -0.08,
            f'Caption: "{caption}"\n'
            f"Annotated categories: {category_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            wrap=True,
        )

    fig.suptitle(
        "COCO 2017 Validation: Qualitative Dataset Audit",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / "qualitative_audit.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()