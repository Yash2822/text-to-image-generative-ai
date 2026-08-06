from pathlib import Path

from src.data.coco_annotations import COCOAnnotationLoader
from src.data.coco_visualization import COCOVisualizer


def main():
    coco_root = (
        Path.home()
        / "fiftyone"
        / "coco-2017"
    )

    raw_dir = coco_root / "raw"

    loader = COCOAnnotationLoader(
        instances_path=raw_dir / "instances_val2017.json",
        captions_path=raw_dir / "captions_val2017.json",
    )

    visualizer = COCOVisualizer(
        loader=loader,
        output_dir="outputs/task1/figures",
    )

    plots = [
        visualizer.plot_category_distribution(),
        visualizer.plot_objects_per_image(),
        visualizer.plot_caption_lengths(),
        visualizer.plot_aspect_ratios(),
        visualizer.plot_orientation_distribution(),
    ]

    print("=== GENERATED FIGURES ===")

    for path in plots:
        print(path)


if __name__ == "__main__":
    main()