from pathlib import Path

import matplotlib.pyplot as plt

from src.data.coco_annotations import COCOAnnotationLoader
from src.data.coco_statistics import COCOStatistics


class COCOVisualizer:
    """Creates exploratory visualizations for COCO."""

    def __init__(
        self,
        loader: COCOAnnotationLoader,
        output_dir: str | Path,
    ) -> None:
        self.loader = loader
        self.statistics = COCOStatistics(loader)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_category_distribution(self) -> Path:
        frequencies = self.statistics.category_frequencies()

        categories = list(frequencies.keys())
        counts = list(frequencies.values())

        fig, ax = plt.subplots(figsize=(10, 16))

        ax.barh(categories[::-1], counts[::-1])

        ax.set_title("COCO 2017 Validation: Object Category Distribution")
        ax.set_xlabel("Number of object annotations")
        ax.set_ylabel("Category")

        fig.tight_layout()

        path = self.output_dir / "category_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    def plot_objects_per_image(self) -> Path:
        counts = [
            len(self.loader.get_annotations(image_id))
            for image_id in self.loader.images_by_id
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(counts, bins=30)

        ax.set_title("Objects per Image")
        ax.set_xlabel("Number of object annotations")
        ax.set_ylabel("Number of images")

        fig.tight_layout()

        path = self.output_dir / "objects_per_image.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    def plot_caption_lengths(self) -> Path:
        lengths = [
            len(annotation["caption"].split())
            for annotation in self.loader.captions["annotations"]
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(lengths, bins=30)

        ax.set_title("Caption Length Distribution")
        ax.set_xlabel("Caption length (words)")
        ax.set_ylabel("Number of captions")

        fig.tight_layout()

        path = self.output_dir / "caption_lengths.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    def plot_aspect_ratios(self) -> Path:
        ratios = [
            image["width"] / image["height"]
            for image in self.loader.instances["images"]
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(ratios, bins=30)
        ax.axvline(
            1.0,
            linestyle="--",
            label="Square (1:1)",
        )

        ax.set_title("Image Aspect Ratio Distribution")
        ax.set_xlabel("Width / Height")
        ax.set_ylabel("Number of images")
        ax.legend()

        fig.tight_layout()

        path = self.output_dir / "aspect_ratios.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    def plot_orientation_distribution(self) -> Path:
        orientations = self.statistics.orientation_counts()

        labels = list(orientations.keys())
        counts = list(orientations.values())

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.bar(labels, counts)

        ax.set_title("Image Orientation Distribution")
        ax.set_xlabel("Orientation")
        ax.set_ylabel("Number of images")

        fig.tight_layout()

        path = self.output_dir / "orientation_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path