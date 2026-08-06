from pathlib import Path

from src.data.coco_annotations import COCOAnnotationLoader
from src.data.coco_sampling import COCOSampler


def main():
    coco_root = Path.home() / "fiftyone" / "coco-2017"
    raw_dir = coco_root / "raw"

    loader = COCOAnnotationLoader(
        instances_path=raw_dir / "instances_val2017.json",
        captions_path=raw_dir / "captions_val2017.json",
    )

    sampler = COCOSampler(loader)

    print("=== REPRESENTATIVE COCO SAMPLES ===")

    for sample in sampler.representative_samples():
        image = loader.images_by_id[sample.image_id]
        annotations = loader.get_annotations(sample.image_id)
        captions = loader.get_captions(sample.image_id)

        print("\n" + "=" * 70)
        print(f"Reason: {sample.reason}")
        print(f"Image ID: {sample.image_id}")
        print(f"File: {image['file_name']}")
        print(f"Size: {image['width']}x{image['height']}")
        print(
            f"Aspect ratio: "
            f"{image['width'] / image['height']:.2f}"
        )
        print(f"Objects: {len(annotations)}")

        print("\nCaptions:")

        for caption in captions:
            print(f"- {caption}")

        print("\nObject categories:")

        categories = [
            loader.get_category_name(annotation["category_id"])
            for annotation in annotations
        ]

        if categories:
            print(", ".join(categories))
        else:
            print("None")


if __name__ == "__main__":
    main()