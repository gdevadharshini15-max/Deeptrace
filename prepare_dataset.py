from pathlib import Path
import shutil
import random

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "dataset" / "faces"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

random.seed(42)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def split_by_video(label):

    source = SOURCE_DIR / label
    output = OUTPUT_DIR

    # Each subfolder represents one original video
    video_folders = [
        folder for folder in source.iterdir()
        if folder.is_dir()
    ]

    random.shuffle(video_folders)

    total_videos = len(video_folders)

    train_end = int(total_videos * TRAIN_RATIO)
    val_end = train_end + int(total_videos * VAL_RATIO)

    train_videos = video_folders[:train_end]
    val_videos = video_folders[train_end:val_end]
    test_videos = video_folders[val_end:]

    splits = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos
    }

    print(f"\n{label.upper()}")
    print(f"Videos: {total_videos}")

    for split, folders in splits.items():

        destination = output / split / label
        destination.mkdir(parents=True, exist_ok=True)

        copied = 0

        for video_folder in folders:

            # Keep each video's folder structure
            target_folder = destination / video_folder.name
            target_folder.mkdir(parents=True, exist_ok=True)

            for image in video_folder.glob("*.jpg"):
                shutil.copy2(
                    image,
                    target_folder / image.name
                )
                copied += 1

        print(
            f"{split.upper()}: "
            f"{len(folders)} videos, "
            f"{copied} face images"
        )


# Process both classes
split_by_video("real")
split_by_video("fake")

print("\nCorrect video-level dataset split completed!")