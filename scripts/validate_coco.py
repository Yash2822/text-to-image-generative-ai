from pathlib import Path

from src.data.coco_annotations import COCOAnnotationLoader
from src.data.coco_validation import COCOValidator


def main():
    coco_root = (
        Path.home()
        / "fiftyone"
        / "coco-2017"
    )

    raw_dir = coco_root / "raw"
    image_dir = coco_root / "validation" / "data"

    loader = COCOAnnotationLoader(
        instances_path=raw_dir / "instances_val2017.json",
        captions_path=raw_dir / "captions_val2017.json",
    )

    validator = COCOValidator(
        loader=loader,
        image_dir=image_dir,
    )

    report = validator.validate()

    print("=== COCO DATASET VALIDATION ===")
    print(f"Valid: {report.is_valid}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"- {error}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()