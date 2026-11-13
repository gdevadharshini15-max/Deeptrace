import sys
from pathlib import Path
import json
import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

PREDICTION_FILE = (
    RESULTS_DIR / "video_predictions.json"
)

TEMPORAL_FILE = (
    RESULTS_DIR
    / "temporal_analysis"
    / "temporal_analysis.json"
)

SUSPICIOUS_DIR = (
    RESULTS_DIR / "suspicious_frames"
)

OUTPUT_DIR = (
    RESULTS_DIR / "forensic_fusion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not PREDICTION_FILE.exists():

    print(
        "ERROR: video_predictions.json not found."
    )

    print(
        "Run predict_video.py first."
    )

    sys.exit(1)


if not TEMPORAL_FILE.exists():

    print(
        "ERROR: temporal_analysis.json not found."
    )

    print(
        "Run temporal_analysis.py first."
    )

    sys.exit(1)


# ============================================================
# LOAD CNN PREDICTION
# ============================================================

with open(
    PREDICTION_FILE,
    "r",
    encoding="utf-8"
) as file:

    prediction_data = json.load(file)


video_name = prediction_data["video"]

cnn_fake_probability = float(
    prediction_data[
        "average_fake_probability"
    ]
)

cnn_confidence = float(
    prediction_data["confidence"]
)


# ============================================================
# LOAD TEMPORAL ANALYSIS
# ============================================================

with open(
    TEMPORAL_FILE,
    "r",
    encoding="utf-8"
) as file:

    temporal_data = json.load(file)


temporal_average = float(
    temporal_data[
        "average_fake_probability"
    ]
)


# ============================================================
# FORENSIC ANALYSIS FUNCTIONS
# ============================================================


def analyze_boundary(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_pixels = np.sum(
        edges > 0
    )

    total_pixels = (
        edges.shape[0]
        * edges.shape[1]
    )

    return float(
        edge_pixels / total_pixels
    )


def analyze_texture(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(
        laplacian.var()
    )


def analyze_frequency(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_float = np.float32(
        gray
    )

    fft = np.fft.fft2(
        gray_float
    )

    fft_shift = np.fft.fftshift(
        fft
    )

    magnitude = np.abs(
        fft_shift
    )

    magnitude_spectrum = np.log1p(
        magnitude
    )

    return float(
        np.mean(magnitude_spectrum)
    )


# ============================================================
# FIND CURRENT SUSPICIOUS FRAMES
# ============================================================

image_files = sorted(
    SUSPICIOUS_DIR.glob("*.jpg")
)


if len(image_files) == 0:

    print(
        "ERROR: No suspicious frames found."
    )

    print(
        "Run predict_video.py first."
    )

    sys.exit(1)


# ============================================================
# CALCULATE FORENSIC FEATURES
# ============================================================

boundary_scores = {}
texture_scores = {}
frequency_scores = {}


for image_path in image_files:

    boundary = analyze_boundary(
        image_path
    )

    texture = analyze_texture(
        image_path
    )

    frequency = analyze_frequency(
        image_path
    )

    if (
        boundary is None
        or texture is None
        or frequency is None
    ):
        continue

    boundary_scores[
        image_path.name
    ] = boundary

    texture_scores[
        image_path.name
    ] = texture

    frequency_scores[
        image_path.name
    ] = frequency


if len(boundary_scores) == 0:

    print(
        "ERROR: Could not analyze suspicious frames."
    )

    sys.exit(1)


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================


def min_max_normalize(values):

    values = np.array(
        values,
        dtype=float
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return np.zeros_like(
            values
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


frame_names = list(
    boundary_scores.keys()
)


boundary_values = [
    boundary_scores[name]
    for name in frame_names
]

texture_values = [
    texture_scores[name]
    for name in frame_names
]

frequency_values = [
    frequency_scores[name]
    for name in frame_names
]


boundary_norm = min_max_normalize(
    boundary_values
)

texture_norm = min_max_normalize(
    texture_values
)

frequency_norm = min_max_normalize(
    frequency_values
)


# ============================================================
# FRAME-LEVEL FORENSIC EVIDENCE
# ============================================================

frame_evidence = {}


for i, frame_name in enumerate(
    frame_names
):

    forensic_score = (

        0.40
        * boundary_norm[i]

        +

        0.30
        * texture_norm[i]

        +

        0.30
        * frequency_norm[i]
    )


    frame_evidence[
        frame_name
    ] = {

        "boundary_score":
            float(
                boundary_scores[
                    frame_name
                ]
            ),

        "texture_score":
            float(
                texture_scores[
                    frame_name
                ]
            ),

        "frequency_score":
            float(
                frequency_scores[
                    frame_name
                ]
            ),

        "normalized_boundary":
            float(
                boundary_norm[i]
            ),

        "normalized_texture":
            float(
                texture_norm[i]
            ),

        "normalized_frequency":
            float(
                frequency_norm[i]
            ),

        "forensic_evidence_score":
            float(
                forensic_score
            )
    }


# ============================================================
# VIDEO-LEVEL FORENSIC SCORE
# ============================================================

average_forensic_score = float(
    np.mean(
        [
            data[
                "forensic_evidence_score"
            ]

            for data
            in frame_evidence.values()
        ]
    )
)


# ============================================================
# EVIDENCE FUSION
# ============================================================

combined_score = (

    0.70
    * cnn_fake_probability

    +

    0.30
    * average_forensic_score
)


# ============================================================
# FINAL ASSESSMENT
# ============================================================

if combined_score >= 0.50:

    final_assessment = "DEEPFAKE"

else:

    final_assessment = "REAL"


# ============================================================
# CREATE REPORT DATA
# ============================================================

report = {

    "video":
        video_name,

    "cnn": {

        "fake_probability":
            cnn_fake_probability,

        "confidence":
            cnn_confidence
    },

    "temporal": {

        "average_fake_probability":
            temporal_average
    },

    "frame_level_forensic_evidence":
        frame_evidence,

    "average_forensic_evidence_score":
        average_forensic_score,

    "combined_evidence_score":
        float(combined_score),

    "decision_threshold":
        0.50,

    "final_assessment":
        final_assessment
}


# ============================================================
# SAVE JSON
# ============================================================

json_path = (
    OUTPUT_DIR
    / f"{Path(video_name).stem}_forensic_evidence.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "       DEEPTRACE EVIDENCE FUSION"
)

print(
    "=========================================="
)

print(
    f"\nVideo: {video_name}"
)

print(
    f"CNN fake probability: "
    f"{cnn_fake_probability:.4f}"
)

print(
    f"Temporal average: "
    f"{temporal_average:.4f}"
)

print(
    f"Average forensic evidence score: "
    f"{average_forensic_score:.4f}"
)

print(
    f"Combined evidence score: "
    f"{combined_score:.4f}"
)

print(
    f"\nFinal assessment: "
    f"{final_assessment}"
)

print(
    "\nFrame-level evidence:"
)

for frame, data in frame_evidence.items():

    print(
        f"{frame} -> "
        f"{data['forensic_evidence_score']:.4f}"
    )


print(
    "\nJSON saved to:"
)

print(
    json_path
)

print(
    "\nEvidence fusion completed!"
)