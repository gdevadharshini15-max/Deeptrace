import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "temporal_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT
# ============================================================

# JSON file containing frame-level predictions
INPUT_FILE = RESULTS_DIR / "video_predictions.json"


# ============================================================
# LOAD FRAME PREDICTIONS
# ============================================================

if not INPUT_FILE.exists():

    print("ERROR: video_predictions.json not found.")

    print(
        "\nRun predict_video.py first "
        "so that frame-level predictions are available."
    )

    sys.exit(1)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


# ============================================================
# EXTRACT PROBABILITIES
# ============================================================

frame_results = data.get(
    "frame_results",
    []
)


if len(frame_results) == 0:

    print(
        "ERROR: No frame predictions found."
    )

    sys.exit(1)


frame_numbers = [
    item["frame_number"]
    for item in frame_results
]


probabilities = [
    item["probability"]
    for item in frame_results
]


probabilities = np.array(
    probabilities,
    dtype=float
)


# ============================================================
# TEMPORAL STATISTICS
# ============================================================

average_probability = np.mean(
    probabilities
)

maximum_probability = np.max(
    probabilities
)

minimum_probability = np.min(
    probabilities
)

std_probability = np.std(
    probabilities
)


# Number of frames whose fake probability
# is above the decision threshold

threshold = 0.5

high_probability_frames = np.sum(
    probabilities >= threshold
)


# ============================================================
# TEMPORAL CONSISTENCY
# ============================================================

if len(probabilities) > 1:

    frame_changes = np.abs(
        np.diff(probabilities)
    )

    average_frame_change = np.mean(
        frame_changes
    )

else:

    average_frame_change = 0.0


# ============================================================
# VIDEO-LEVEL TEMPORAL ASSESSMENT
# ============================================================

if average_probability >= threshold:

    temporal_assessment = "DEEPFAKE"

else:

    temporal_assessment = "REAL"


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==========================================")
print("       DEEPTRACE TEMPORAL ANALYSIS")
print("==========================================")

print(
    f"\nFrames analysed: "
    f"{len(probabilities)}"
)

print(
    f"Average fake probability: "
    f"{average_probability:.4f}"
)

print(
    f"Maximum fake probability: "
    f"{maximum_probability:.4f}"
)

print(
    f"Minimum fake probability: "
    f"{minimum_probability:.4f}"
)

print(
    f"Probability standard deviation: "
    f"{std_probability:.4f}"
)

print(
    f"Frames >= 0.5: "
    f"{high_probability_frames}"
)

print(
    f"Average frame-to-frame change: "
    f"{average_frame_change:.4f}"
)

print(
    f"Temporal assessment: "
    f"{temporal_assessment}"
)


# ============================================================
# SAVE TEMPORAL JSON
# ============================================================

temporal_result = {

    "video": data.get(
        "video",
        "unknown"
    ),

    "number_of_frames": int(
        len(probabilities)
    ),

    "average_fake_probability":
        float(average_probability),

    "maximum_fake_probability":
        float(maximum_probability),

    "minimum_fake_probability":
        float(minimum_probability),

    "probability_std":
        float(std_probability),

    "frames_above_threshold":
        int(high_probability_frames),

    "average_frame_to_frame_change":
        float(average_frame_change),

    "decision_threshold":
        threshold,

    "temporal_assessment":
        temporal_assessment,

    "frame_probabilities": [
        {
            "frame_number": int(
                frame_numbers[i]
            ),
            "fake_probability": float(
                probabilities[i]
            )
        }
        for i in range(
            len(probabilities)
        )
    ]
}


JSON_OUTPUT = (
    OUTPUT_DIR /
    "temporal_analysis.json"
)


with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        temporal_result,
        file,
        indent=4
    )


# ============================================================
# CREATE TEMPORAL GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    frame_numbers,
    probabilities,
    marker="o"
)

plt.axhline(
    y=0.5,
    linestyle="--"
)

plt.xlabel(
    "Frame Number"
)

plt.ylabel(
    "Fake Probability"
)

plt.title(
    "DeepTrace Temporal Fake Probability Analysis"
)

plt.ylim(
    0,
    1
)

plt.grid(
    True,
    alpha=0.3
)

GRAPH_OUTPUT = (
    OUTPUT_DIR /
    "temporal_probability.png"
)

plt.savefig(
    GRAPH_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print("\nTemporal analysis completed!")

print(
    "\nJSON saved to:"
)

print(
    JSON_OUTPUT
)

print(
    "\nGraph saved to:"
)

print(
    GRAPH_OUTPUT
)