from dataclasses import dataclass

from src.data.coco_annotations import COCOAnnotationLoader


@dataclass(frozen=True)
class SampledImage:
    image_id: int
    reason: str


class COCOSampler:
    """Selects representative COCO images for qualitative analysis."""

    def __init__(self, loader: COCOAnnotationLoader) -> None:
        self.loader = loader

    def _object_count(self, image_id: int) -> int:
        return len(self.loader.get_annotations(image_id))

    def simplest_image(self) -> SampledImage:
        candidates = [
            image_id
            for image_id in self.loader.images_by_id
            if self._object_count(image_id) > 0
        ]

        image_id = min(
            candidates,
            key=self._object_count,
        )

        return SampledImage(
            image_id=image_id,
            reason="Simple scene with few annotated objects",
        )

    def most_complex_image(self) -> SampledImage:
        image_id = max(
            self.loader.images_by_id,
            key=self._object_count,
        )

        return SampledImage(
            image_id=image_id,
            reason="Complex scene with many annotated objects",
        )

    def zero_object_image(self) -> SampledImage:
        for image_id in self.loader.images_by_id:
            if self._object_count(image_id) == 0:
                return SampledImage(
                    image_id=image_id,
                    reason="Image with captions but no object annotations",
                )

        raise ValueError("No zero-object images found")

    def portrait_image(self) -> SampledImage:
        candidates = []

        for image_id, image in self.loader.images_by_id.items():
            ratio = image["width"] / image["height"]

            if ratio < 0.8:
                candidates.append((ratio, image_id))

        if not candidates:
            raise ValueError("No portrait images found")

        _, image_id = min(candidates)

        return SampledImage(
            image_id=image_id,
            reason="Strong portrait-oriented example",
        )

    def landscape_image(self) -> SampledImage:
        candidates = []

        for image_id, image in self.loader.images_by_id.items():
            ratio = image["width"] / image["height"]

            if ratio > 1.5:
                candidates.append((ratio, image_id))

        if not candidates:
            raise ValueError("No landscape images found")

        _, image_id = max(candidates)

        return SampledImage(
            image_id=image_id,
            reason="Strong landscape-oriented example",
        )

    def representative_samples(self) -> list[SampledImage]:
        return [
            self.simplest_image(),
            self.most_complex_image(),
            self.zero_object_image(),
            self.portrait_image(),
            self.landscape_image(),
        ]