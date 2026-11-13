from pathlib import Path
import cv2
import torch

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "dataset" / "processed"
OUTPUT_DIR = BASE_DIR / "dataset" / "normalized"

# Image size
IMAGE_SIZE = 224


def normalize_images(split, label):

    source = SOURCE_DIR / split / label
    destination = OUTPUT_DIR / split / label

    destination.mkdir(parents=True, exist_ok=True)

    images = list(source.rglob("*.jpg"))

    print(f"\n{split.upper()} - {label.upper()}")
    print(f"Images found: {len(images)}")

    for index, image_path in enumerate(images, start=1):

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # OpenCV uses BGR → convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure 224 × 224
        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert 0-255 → 0.0-1.0
        tensor = torch.from_numpy(image).float() / 255.0

        # HWC → CHW
        tensor = tensor.permute(2, 0, 1)

        # Save tensor
        relative_path = image_path.relative_to(source)
        output_path = destination / relative_path.with_suffix(".pt")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(tensor, output_path)

        if index % 100 == 0:
            print(f"Processed: {index}/{len(images)}")


# Process train / validation / test
for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        normalize_images(split, label)


print("\nNormalization completed successfully!")