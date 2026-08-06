import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class COCOAnnotationLoader:
    """Loads and indexes COCO instance and caption annotations."""

    def __init__(
        self,
        instances_path: str | Path,
        captions_path: str | Path,
    ) -> None:
        self.instances_path = Path(instances_path)
        self.captions_path = Path(captions_path)

        self._validate_paths()

        self.instances = self._load_json(self.instances_path)
        self.captions = self._load_json(self.captions_path)

        self.images_by_id = self._build_image_index()
        self.categories_by_id = self._build_category_index()
        self.annotations_by_image = self._build_annotation_index()
        self.captions_by_image = self._build_caption_index()

        self._validate_dataset()

    def _validate_paths(self) -> None:
        for path in (self.instances_path, self.captions_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"COCO annotation file not found: {path}"
                )

            if path.suffix.lower() != ".json":
                raise ValueError(
                    f"Expected a JSON annotation file, got: {path}"
                )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _build_image_index(self) -> dict[int, dict[str, Any]]:
        return {
            image["id"]: image
            for image in self.instances["images"]
        }

    def _build_category_index(self) -> dict[int, str]:
        return {
            category["id"]: category["name"]
            for category in self.instances["categories"]
        }

    def _build_annotation_index(
        self,
    ) -> dict[int, list[dict[str, Any]]]:
        index = defaultdict(list)

        for annotation in self.instances["annotations"]:
            index[annotation["image_id"]].append(annotation)

        return dict(index)

    def _build_caption_index(self) -> dict[int, list[str]]:
        index = defaultdict(list)

        for annotation in self.captions["annotations"]:
            index[annotation["image_id"]].append(
                annotation["caption"]
            )

        return dict(index)

    def _validate_dataset(self) -> None:
        instance_image_ids = set(self.images_by_id)

        caption_image_ids = {
            image["id"]
            for image in self.captions["images"]
        }

        if instance_image_ids != caption_image_ids:
            raise ValueError(
                "Instance and caption files reference different image sets"
            )

    @property
    def num_images(self) -> int:
        return len(self.images_by_id)

    @property
    def num_categories(self) -> int:
        return len(self.categories_by_id)

    @property
    def num_annotations(self) -> int:
        return len(self.instances["annotations"])

    @property
    def num_captions(self) -> int:
        return len(self.captions["annotations"])

    def get_image(self, image_id: int) -> dict[str, Any]:
        if image_id not in self.images_by_id:
            raise KeyError(f"Unknown COCO image ID: {image_id}")

        return self.images_by_id[image_id]

    def get_captions(self, image_id: int) -> list[str]:
        return self.captions_by_image.get(image_id, [])

    def get_annotations(
        self,
        image_id: int,
    ) -> list[dict[str, Any]]:
        return self.annotations_by_image.get(image_id, [])

    def get_category_name(self, category_id: int) -> str:
        if category_id not in self.categories_by_id:
            raise KeyError(
                f"Unknown COCO category ID: {category_id}"
            )

        return self.categories_by_id[category_id]