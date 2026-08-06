from pathlib import Path

from src.data.coco_annotations import COCOAnnotationLoader
from src.data.coco_statistics import COCOStatistics


def print_stats(name, stats):
    print(f"\n{name}")
    print(f"  Mean:   {stats.mean:.2f}")
    print(f"  Median: {stats.median:.2f}")
    print(f"  Min:    {stats.minimum:.2f}")
    print(f"  Max:    {stats.maximum:.2f}")
    print(f"  Std:    {stats.std:.2f}")
    print(f"  P25:    {stats.p25:.2f}")
    print(f"  P75:    {stats.p75:.2f}")
    print(f"  P95:    {stats.p95:.2f}")


def main():
    raw_dir = (
        Path.home()
        / "fiftyone"
        / "coco-2017"
        / "raw"
    )

    loader = COCOAnnotationLoader(
        instances_path=raw_dir / "instances_val2017.json",
        captions_path=raw_dir / "captions_val2017.json",
    )

    statistics = COCOStatistics(loader)

    print("=== DATASET SUMMARY ===")

    for name, value in statistics.dataset_summary().items():
        print(f"{name.capitalize()}: {value}")

    print("\n=== TOP 15 CATEGORIES ===")

    frequencies = statistics.category_frequencies()

    for category, count in list(frequencies.items())[:15]:
        print(f"{category:<20} {count}")

    print_stats(
        "Objects per image",
        statistics.objects_per_image(),
    )

    print_stats(
        "Captions per image",
        statistics.captions_per_image(),
    )

    print_stats(
        "Caption length (words)",
        statistics.caption_word_lengths(),
    )

    print_stats(
        "Image width",
        statistics.image_widths(),
    )

    print_stats(
        "Image height",
        statistics.image_heights(),
    )

    print_stats(
        "Aspect ratio",
        statistics.aspect_ratios(),
    )

    print("\nImage orientations:")

    for orientation, count in statistics.orientation_counts().items():
        print(f"  {orientation.capitalize():<10} {count}")

    print("\n=== BOTTOM 15 CATEGORIES ===")

    for category, count in statistics.least_common_categories():
        print(f"{category:<20} {count}")

    print(
        "\nImages without object annotations:",
    statistics.images_without_objects(),
    )

    print(
        "Category imbalance ratio:",
        f"{statistics.category_imbalance_ratio():.2f}:1",
    )


if __name__ == "__main__":
    main()