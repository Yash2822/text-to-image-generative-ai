import fiftyone as fo
import fiftyone.zoo as foz


def main():
    print("FiftyOne version:", fo.__version__)
    print("Starting COCO 2017 validation acquisition...")

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        dataset_name="coco-2017-validation",
    )
    dataset.persistent = True

    print("\nDataset acquisition complete.")
    print("Dataset name:", dataset.name)
    print("Number of samples:", len(dataset))
    print("Persistent:", dataset.persistent)


if __name__ == "__main__":
    main()