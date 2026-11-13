import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import torch
import numpy as np
import mediapipe as mp

from models.cnn_model import DeepTraceCNN


# ============================================================
# VIDEO INPUT
# ============================================================

if len(sys.argv) < 2:

    print("Usage:")
    print("python evaluation/predict_video.py <video_path>")

    sys.exit(1)


VIDEO_PATH = Path(sys.argv[1])

if not VIDEO_PATH.is_absolute():
    VIDEO_PATH = BASE_DIR / VIDEO_PATH


if not VIDEO_PATH.exists():

    print(
        f"ERROR: Video not found:\n{VIDEO_PATH}"
    )

    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "deeptrace_cnn_final.pth"
)

RESULTS_DIR = BASE_DIR / "results"

SUSPICIOUS_DIR = (
    RESULTS_DIR
    / "suspicious_frames"
)

SUSPICIOUS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)
print("Video:", VIDEO_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

model = DeepTraceCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# FACE DETECTION
# ============================================================

mp_face_detection = (
    mp.solutions.face_detection
)


# ============================================================
# VIDEO SETTINGS
# ============================================================

FRAME_INTERVAL = 10
MAX_FRAMES = 30

frame_number = 0
processed_frames = 0

predictions = []
probabilities = []

frame_results = []


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    print(
        "ERROR: Could not open video."
    )

    sys.exit(1)


# ============================================================
# PROCESS VIDEO
# ============================================================

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
) as detector:

    while (
        cap.isOpened()
        and processed_frames < MAX_FRAMES
    ):

        success, frame = cap.read()

        if not success:
            break

        # --------------------------------------
        # SAMPLE EVERY 10TH FRAME
        # --------------------------------------

        if frame_number % FRAME_INTERVAL != 0:

            frame_number += 1
            continue


        # --------------------------------------
        # BGR -> RGB
        # --------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # --------------------------------------
        # FACE DETECTION
        # --------------------------------------

        results = detector.process(
            rgb_frame
        )

        if not results.detections:

            frame_number += 1
            continue


        # --------------------------------------
        # FIRST DETECTED FACE
        # --------------------------------------

        detection = results.detections[0]

        height, width, _ = frame.shape

        box = (
            detection
            .location_data
            .relative_bounding_box
        )


        x = max(
            0,
            int(box.xmin * width)
        )

        y = max(
            0,
            int(box.ymin * height)
        )

        w = int(
            box.width * width
        )

        h = int(
            box.height * height
        )

        x2 = min(
            width,
            x + w
        )

        y2 = min(
            height,
            y + h
        )


        if x2 <= x or y2 <= y:

            frame_number += 1
            continue


        # --------------------------------------
        # CROP FACE
        # --------------------------------------

        face = frame[
            y:y2,
            x:x2
        ]


        # --------------------------------------
        # RESIZE
        # --------------------------------------

        face = cv2.resize(
            face,
            (224, 224)
        )


        # --------------------------------------
        # BGR -> RGB
        # --------------------------------------

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )


        # Keep original RGB face
        original_face = face.copy()


        # --------------------------------------
        # NORMALIZE
        # --------------------------------------

        face = (
            face.astype(np.float32)
            / 255.0
        )


        # --------------------------------------
        # HWC -> CHW
        # --------------------------------------

        face = np.transpose(
            face,
            (2, 0, 1)
        )


        # --------------------------------------
        # NUMPY -> PYTORCH
        # --------------------------------------

        face_tensor = torch.from_numpy(
            face
        ).float()


        face_tensor = (
            face_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )


        # --------------------------------------
        # CNN PREDICTION
        # --------------------------------------

        with torch.no_grad():

            output = model(
                face_tensor
            )

            fake_probability = (
                torch.sigmoid(output)
                .item()
            )


        prediction = (
            1
            if fake_probability >= 0.5
            else 0
        )


        # --------------------------------------
        # STORE RESULTS
        # --------------------------------------

        predictions.append(
            prediction
        )

        probabilities.append(
            fake_probability
        )

        frame_results.append(
            {
                "frame_number": int(
                    frame_number
                ),

                "probability": float(
                    fake_probability
                ),

                "prediction": (
                    "DEEPFAKE"
                    if prediction == 1
                    else "REAL"
                ),

                "face": original_face
            }
        )


        processed_frames += 1


        label = (
            "DEEPFAKE"
            if prediction == 1
            else "REAL"
        )


        print(
            f"Frame {frame_number:04d} -> "
            f"{label} | "
            f"Fake probability: "
            f"{fake_probability:.4f}"
        )


        frame_number += 1


cap.release()


# ============================================================
# CHECK RESULTS
# ============================================================

if len(predictions) == 0:

    print(
        "\nNo valid face frames were detected."
    )

    sys.exit(1)


# ============================================================
# VIDEO-LEVEL PREDICTION
# ============================================================

average_probability = np.mean(
    probabilities
)


if average_probability >= 0.5:

    final_prediction = "DEEPFAKE"

    confidence = average_probability

else:

    final_prediction = "REAL"

    confidence = (
        1 - average_probability
    )


# ============================================================
# SAVE FRAME PREDICTIONS JSON
# ============================================================

json_frame_results = []

for result in frame_results:

    json_frame_results.append(
        {
            "frame_number":
                result["frame_number"],

            "probability":
                result["probability"],

            "prediction":
                result["prediction"]
        }
    )


prediction_data = {

    "video":
        VIDEO_PATH.name,

    "frames_analysed":
        len(predictions),

    "average_fake_probability":
        float(average_probability),

    "final_prediction":
        final_prediction,

    "confidence":
        float(confidence),

    "frame_results":
        json_frame_results
}


PREDICTION_JSON = (
    RESULTS_DIR
    / "video_predictions.json"
)


with open(
    PREDICTION_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        prediction_data,
        file,
        indent=4
    )


# ============================================================
# SAVE TOP 3 HIGHEST-SCORING FRAMES
# ============================================================

frame_results.sort(
    key=lambda item:
    item["probability"],
    reverse=True
)

top_frames = frame_results[:3]


print(
    "\nSaving highest-scoring frames..."
)


# Remove previous suspicious frames
for old_file in SUSPICIOUS_DIR.glob("*.jpg"):

    old_file.unlink()


for rank, result in enumerate(
    top_frames,
    start=1
):

    output_path = (
        SUSPICIOUS_DIR
        / f"{VIDEO_PATH.stem}_rank_"
          f"{rank}_frame_"
          f"{result['frame_number']:04d}.jpg"
    )


    face_image = result["face"]


    # RGB -> BGR
    face_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_RGB2BGR
    )


    cv2.imwrite(
        str(output_path),
        face_image
    )


    print(
        f"Saved top frame {rank}: "
        f"{output_path.name} | "
        f"Fake probability: "
        f"{result['probability']:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "       DEEPTRACE VIDEO ANALYSIS"
)

print(
    "=========================================="
)

print(
    f"Video: {VIDEO_PATH.name}"
)

print(
    f"Frames analysed: "
    f"{len(predictions)}"
)

print(
    f"Average fake probability: "
    f"{average_probability:.4f}"
)

print(
    f"Final prediction: "
    f"{final_prediction}"
)

print(
    f"Confidence: "
    f"{confidence * 100:.2f}%"
)

print(
    "\nFrame prediction JSON:"
)

print(
    PREDICTION_JSON
)

print(
    "\nSuspicious frames:"
)

print(
    SUSPICIOUS_DIR
)

print(
    "\n=========================================="
)