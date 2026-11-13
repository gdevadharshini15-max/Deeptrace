from pathlib import Path
import cv2

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
OUTPUT_DIR = BASE_DIR / "dataset" / "extracted"

# Extract every 10th frame
FRAME_INTERVAL = 10

# Maximum frames from each video
MAX_FRAMES = 30


def extract_frames(input_dir, output_dir, label):
    videos = list(input_dir.glob("*.mp4"))

    print(f"\nProcessing {label.upper()} videos: {len(videos)}")

    for video_index, video_path in enumerate(videos, start=1):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open: {video_path.name}")
            continue

        video_name = video_path.stem
        video_output = output_dir / video_name
        video_output.mkdir(parents=True, exist_ok=True)

        frame_number = 0
        saved_frames = 0

        while cap.isOpened() and saved_frames < MAX_FRAMES:

            success, frame = cap.read()

            if not success:
                break

            if frame_number % FRAME_INTERVAL == 0:

                frame_name = f"frame_{saved_frames:04d}.jpg"
                frame_path = video_output / frame_name

                cv2.imwrite(str(frame_path), frame)

                saved_frames += 1

            frame_number += 1

        cap.release()

        print(
            f"[{video_index}/{len(videos)}] "
            f"{video_path.name} → {saved_frames} frames"
        )


# REAL videos
extract_frames(
    RAW_DIR / "real",
    OUTPUT_DIR / "real",
    "real"
)

# FAKE videos
extract_frames(
    RAW_DIR / "fake",
    OUTPUT_DIR / "fake",
    "fake"
)

print("\nFrame extraction completed!")