from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.data.coco_annotations import COCOAnnotationLoader


@dataclass(frozen=True)
class SummaryStatistics:
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float
    p25: float
    p75: float
    p95: float


class COCOStatistics:
    """Computes descriptive statistics for COCO annotations."""

    def __init__(self, loader: COCOAnnotationLoader) -> None:
        self.loader = loader

    @staticmethod
    def _summarize(values: list[float]) -> SummaryStatistics:
        if not values:
            raise ValueError("Cannot summarize an empty collection")

        array = np.asarray(values, dtype=np.float64)

        return SummaryStatistics(
            mean=float(np.mean(array)),
            median=float(np.median(array)),
            minimum=float(np.min(array)),
            maximum=float(np.max(array)),
            std=float(np.std(array)),
            p25=float(np.percentile(array, 25)),
            p75=float(np.percentile(array, 75)),
            p95=float(np.percentile(array, 95)),
    )

    def dataset_summary(self) -> dict[str, int]:
        return {
            "images": self.loader.num_images,
            "categories": self.loader.num_categories,
            "annotations": self.loader.num_annotations,
            "captions": self.loader.num_captions,
        }

    def category_frequencies(self) -> dict[str, int]:
        counts = Counter()

        for annotation in self.loader.instances["annotations"]:
            category_name = self.loader.get_category_name(
                annotation["category_id"]
            )
            counts[category_name] += 1

        return dict(counts.most_common())

    def least_common_categories(
    self,
    n: int = 15,
    ) -> list[tuple[str, int]]:
        frequencies = self.category_frequencies()

        return sorted(
            frequencies.items(),
            key=lambda item: item[1],
        )[:n]

    def category_imbalance_ratio(self) -> float:
        frequencies = list(
            self.category_frequencies().values()
        )

        return max(frequencies) / min(frequencies)

    def images_without_objects(self) -> int:
        return sum(
            1
            for image_id in self.loader.images_by_id
            if not self.loader.get_annotations(image_id)
        )

    def objects_per_image(self) -> SummaryStatistics:
        counts = [
            len(self.loader.get_annotations(image_id))
            for image_id in self.loader.images_by_id
        ]

        return self._summarize(counts)

    def captions_per_image(self) -> SummaryStatistics:
        counts = [
            len(self.loader.get_captions(image_id))
            for image_id in self.loader.images_by_id
        ]

        return self._summarize(counts)

    def caption_word_lengths(self) -> SummaryStatistics:
        lengths = []

        for annotation in self.loader.captions["annotations"]:
            caption = annotation["caption"]
            lengths.append(len(caption.split()))

        return self._summarize(lengths)

    def image_widths(self) -> SummaryStatistics:
        widths = [
            image["width"]
            for image in self.loader.instances["images"]
        ]

        return self._summarize(widths)

    def image_heights(self) -> SummaryStatistics:
        heights = [
            image["height"]
            for image in self.loader.instances["images"]
        ]

        return self._summarize(heights)

    def image_areas(self) -> SummaryStatistics:
        areas = [
            image["width"] * image["height"]
            for image in self.loader.instances["images"]
        ]

        return self._summarize(areas)

    def aspect_ratios(self) -> SummaryStatistics:
        ratios = [
            image["width"] / image["height"]
            for image in self.loader.instances["images"]
        ]

        return self._summarize(ratios)

    def orientation_counts(self) -> dict[str, int]:
        counts = Counter()

        for image in self.loader.instances["images"]:
            ratio = image["width"] / image["height"]

            if 0.95 <= ratio <= 1.05:
                counts["square"] += 1
            elif ratio > 1.05:
                counts["landscape"] += 1
            else:
                counts["portrait"] += 1

        return dict(counts)