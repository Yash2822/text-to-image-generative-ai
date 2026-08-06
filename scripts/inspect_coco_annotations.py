import json
from pathlib import Path


COCO_ROOT = Path.home() / "fiftyone" / "coco-2017" / "raw"

INSTANCES_PATH = COCO_ROOT / "instances_val2017.json"
CAPTIONS_PATH = COCO_ROOT / "captions_val2017.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    instances = load_json(INSTANCES_PATH)
    captions = load_json(CAPTIONS_PATH)

    print("\n=== INSTANCE ANNOTATIONS ===")
    print("Top-level keys:", instances.keys())

    print("\nImages:", len(instances["images"]))
    print("Annotations:", len(instances["annotations"]))
    print("Categories:", len(instances["categories"]))

    print("\nFirst image:")
    print(instances["images"][0])

    print("\nFirst object annotation:")
    print(instances["annotations"][0])

    print("\nFirst category:")
    print(instances["categories"][0])

    print("\n=== CAPTION ANNOTATIONS ===")
    print("Top-level keys:", captions.keys())

    print("\nImages:", len(captions["images"]))
    print("Captions:", len(captions["annotations"]))

    print("\nFirst caption annotation:")
    print(captions["annotations"][0])


if __name__ == "__main__":
    main()