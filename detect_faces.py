from pathlib import Path
import cv2
import mediapipe as mp

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "dataset" / "extracted"
OUTPUT_DIR = BASE_DIR / "dataset" / "faces"

# MediaPipe face detector
mp_face_detection = mp.solutions.face_detection


def process_frames(input_dir, output_dir, label):
    frames = list(input_dir.rglob("*.jpg"))

    print(f"\nProcessing {label.upper()} frames: {len(frames)}")

    saved = 0
    no_face = 0

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as detector:

        for index, frame_path in enumerate(frames, start=1):

            image = cv2.imread(str(frame_path))

            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = detector.process(rgb_image)

            if not results.detections:
                no_face += 1
                continue

            height, width, _ = image.shape

            # Use the first detected face
            detection = results.detections[0]
            box = detection.location_data.relative_bounding_box

            x = max(0, int(box.xmin * width))
            y = max(0, int(box.ymin * height))
            w = int(box.width * width)
            h = int(box.height * height)

            x2 = min(width, x + w)
            y2 = min(height, y + h)

            # Skip invalid boxes
            if x2 <= x or y2 <= y:
                continue

            face = image[y:y2, x:x2]

            # Resize face
            face = cv2.resize(face, (224, 224))

            # Preserve video structure
            relative_path = frame_path.relative_to(input_dir)
            output_path = output_dir / relative_path

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            cv2.imwrite(str(output_path), face)

            saved += 1

            if index % 100 == 0:
                print(
                    f"Processed {index}/{len(frames)} "
                    f"| Faces saved: {saved}"
                )

    print(f"\n{label.upper()} COMPLETE")
    print(f"Faces saved: {saved}")
    print(f"No face detected: {no_face}")


# Process REAL
process_frames(
    INPUT_DIR / "real",
    OUTPUT_DIR / "real",
    "real"
)

# Process FAKE
process_frames(
    INPUT_DIR / "fake",
    OUTPUT_DIR / "fake",
    "fake"
)

print("\nFace detection and cropping completed!")