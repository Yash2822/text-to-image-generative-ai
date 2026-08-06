from dataclasses import dataclass, field
from pathlib import Path

from src.data.coco_annotations import COCOAnnotationLoader


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class COCOValidator:
    """Performs integrity checks on a loaded COCO dataset."""

    def __init__(
        self,
        loader: COCOAnnotationLoader,
        image_dir: str | Path,
    ) -> None:
        self.loader = loader
        self.image_dir = Path(image_dir)

    def validate(self) -> ValidationReport:
        report = ValidationReport()

        self._validate_image_directory(report)
        self._validate_unique_ids(report)
        self._validate_image_files(report)
        self._validate_annotation_references(report)
        self._validate_category_references(report)
        self._validate_caption_references(report)
        self._validate_image_dimensions(report)
        self._validate_bounding_boxes(report)
        self._validate_captions(report)

        return report

    def _validate_image_directory(
        self,
        report: ValidationReport,
    ) -> None:
        if not self.image_dir.exists():
            report.errors.append(
                f"Image directory does not exist: {self.image_dir}"
            )

    def _validate_unique_ids(
        self,
        report: ValidationReport,
    ) -> None:
        image_ids = [
            image["id"]
            for image in self.loader.instances["images"]
        ]

        annotation_ids = [
            annotation["id"]
            for annotation in self.loader.instances["annotations"]
        ]

        caption_ids = [
            annotation["id"]
            for annotation in self.loader.captions["annotations"]
        ]

        if len(image_ids) != len(set(image_ids)):
            report.errors.append("Duplicate image IDs detected")

        if len(annotation_ids) != len(set(annotation_ids)):
            report.errors.append("Duplicate object annotation IDs detected")

        if len(caption_ids) != len(set(caption_ids)):
            report.errors.append("Duplicate caption IDs detected")

    def _validate_image_files(
        self,
        report: ValidationReport,
    ) -> None:
        missing = []

        for image in self.loader.instances["images"]:
            path = self.image_dir / image["file_name"]

            if not path.exists():
                missing.append(image["file_name"])

        if missing:
            report.errors.append(
                f"{len(missing)} image files are missing"
            )

    def _validate_annotation_references(
        self,
        report: ValidationReport,
    ) -> None:
        valid_image_ids = set(self.loader.images_by_id)

        invalid = [
            annotation["id"]
            for annotation in self.loader.instances["annotations"]
            if annotation["image_id"] not in valid_image_ids
        ]

        if invalid:
            report.errors.append(
                f"{len(invalid)} object annotations reference "
                "unknown images"
            )

    def _validate_category_references(
        self,
        report: ValidationReport,
    ) -> None:
        valid_category_ids = set(self.loader.categories_by_id)

        invalid = [
            annotation["id"]
            for annotation in self.loader.instances["annotations"]
            if annotation["category_id"] not in valid_category_ids
        ]

        if invalid:
            report.errors.append(
                f"{len(invalid)} annotations reference "
                "unknown categories"
            )

    def _validate_caption_references(
        self,
        report: ValidationReport,
    ) -> None:
        valid_image_ids = set(self.loader.images_by_id)

        invalid = [
            annotation["id"]
            for annotation in self.loader.captions["annotations"]
            if annotation["image_id"] not in valid_image_ids
        ]

        if invalid:
            report.errors.append(
                f"{len(invalid)} captions reference unknown images"
            )

    def _validate_image_dimensions(
        self,
        report: ValidationReport,
    ) -> None:
        invalid = [
            image["id"]
            for image in self.loader.instances["images"]
            if image["width"] <= 0 or image["height"] <= 0
        ]

        if invalid:
            report.errors.append(
                f"{len(invalid)} images have invalid dimensions"
            )

    def _validate_bounding_boxes(
        self,
        report: ValidationReport,
    ) -> None:
        invalid = 0

        for annotation in self.loader.instances["annotations"]:
            bbox = annotation.get("bbox")

            if not bbox or len(bbox) != 4:
                invalid += 1
                continue

            _, _, width, height = bbox

            if width <= 0 or height <= 0:
                invalid += 1

        if invalid:
            report.errors.append(
                f"{invalid} annotations have invalid bounding boxes"
            )

    def _validate_captions(
        self,
        report: ValidationReport,
    ) -> None:
        empty = [
            annotation["id"]
            for annotation in self.loader.captions["annotations"]
            if not annotation["caption"].strip()
        ]

        if empty:
            report.errors.append(
                f"{len(empty)} empty captions detected"
            )

        images_without_captions = [
            image_id
            for image_id in self.loader.images_by_id
            if not self.loader.get_captions(image_id)
        ]

        if images_without_captions:
            report.warnings.append(
                f"{len(images_without_captions)} images have no captions"
            )