from pathlib import Path
import json
from datetime import datetime
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FUSION_DIR = (
    BASE_DIR
    / "results"
    / "forensic_fusion"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND FORENSIC JSON
# ============================================================

json_files = list(
    FUSION_DIR.glob(
        "*_forensic_evidence.json"
    )
)


if len(json_files) == 0:

    print(
        "ERROR: No forensic evidence JSON found."
    )

    print(
        "Run evidence_fusion.py first."
    )

    sys.exit(1)


# Use the newest generated JSON
INPUT_FILE = max(
    json_files,
    key=lambda path: path.stat().st_mtime
)


# ============================================================
# LOAD EVIDENCE
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    report = json.load(file)


# ============================================================
# EXTRACT INFORMATION
# ============================================================

video_name = report[
    "video"
]

cnn = report[
    "cnn"
]

temporal = report[
    "temporal"
]

frame_evidence = report[
    "frame_level_forensic_evidence"
]

average_forensic_score = report[
    "average_forensic_evidence_score"
]

combined_score = report[
    "combined_evidence_score"
]

decision_threshold = report.get(
    "decision_threshold",
    0.50
)

final_assessment = report[
    "final_assessment"
]


# ============================================================
# FIND HIGHEST FORENSIC FRAME
# ============================================================

highest_frame = max(
    frame_evidence.items(),
    key=lambda item:
    item[1][
        "forensic_evidence_score"
    ]
)

highest_frame_name = (
    highest_frame[0]
)

highest_frame_data = (
    highest_frame[1]
)


# ============================================================
# OUTPUT FILE
# ============================================================

video_stem = Path(
    video_name
).stem

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{video_stem}_forensic_report.txt"
)


# ============================================================
# GENERATE REPORT
# ============================================================

report_text = f"""
============================================================
                 DEEPTRACE FORENSIC REPORT
============================================================

Video Analyzed:
{video_name}

Report Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


============================================================
                    FINAL ASSESSMENT
============================================================

Classification:
{final_assessment}

Combined Evidence Score:
{combined_score:.4f}

Decision Threshold:
{decision_threshold:.4f}


============================================================
                    CNN ANALYSIS
============================================================

CNN Fake Probability:
{cnn["fake_probability"]:.4f}

CNN Fake Probability (%):
{cnn["fake_probability"] * 100:.2f}%

CNN Confidence:
{cnn["confidence"]:.4f}

CNN Confidence (%):
{cnn["confidence"] * 100:.2f}%


============================================================
                  TEMPORAL ANALYSIS
============================================================

Average Fake Probability:
{temporal["average_fake_probability"]:.4f}

Average Fake Probability (%):
{temporal["average_fake_probability"] * 100:.2f}%


============================================================
              FRAME-LEVEL FORENSIC ANALYSIS
============================================================

"""


for frame_name, data in frame_evidence.items():

    report_text += f"""
Frame:
{frame_name}

Boundary Score:
{data["boundary_score"]:.4f}

Texture Score:
{data["texture_score"]:.4f}

Frequency Score:
{data["frequency_score"]:.4f}

Normalized Boundary:
{data["normalized_boundary"]:.4f}

Normalized Texture:
{data["normalized_texture"]:.4f}

Normalized Frequency:
{data["normalized_frequency"]:.4f}

Combined Forensic Evidence Score:
{data["forensic_evidence_score"]:.4f}

------------------------------------------------------------
"""


# ============================================================
# HIGHEST FORENSIC FRAME
# ============================================================

report_text += f"""

============================================================
             HIGHEST FORENSIC-SCORING FRAME
============================================================

Frame:
{highest_frame_name}

Forensic Evidence Score:
{highest_frame_data["forensic_evidence_score"]:.4f}

Note:
This is the highest relative forensic evidence score
among the analyzed candidate frames.

It is NOT a probability of manipulation.


============================================================
                    EVIDENCE FUSION
============================================================

Average Forensic Evidence Score:
{average_forensic_score:.4f}

CNN Fake Probability:
{cnn["fake_probability"]:.4f}

Combined Evidence Score:
{combined_score:.4f}

Decision Threshold:
{decision_threshold:.4f}


============================================================
                      CONCLUSION
============================================================

The DeepTrace system classified the analyzed video as:

{final_assessment}

The CNN produced a fake probability of
{cnn["fake_probability"] * 100:.2f}% for the analyzed video.

Temporal analysis produced an average fake probability of
{temporal["average_fake_probability"] * 100:.2f}%.

Additional forensic evidence was obtained through
boundary, texture and frequency analysis.

The highest relative forensic evidence score was observed
for frame {highest_frame_name}, with a score of
{highest_frame_data["forensic_evidence_score"]:.4f}.

The combined evidence score was
{combined_score:.4f}.

The decision threshold was
{decision_threshold:.4f}.

Therefore, the final DeepTrace assessment is:

{final_assessment}


IMPORTANT:
The forensic feature scores are supporting analytical
measurements. They should not be interpreted individually
as probabilities of deepfake manipulation.

The combined evidence score is an engineering-based
evidence-fusion score and should not be interpreted as
a calibrated probability.


============================================================
                  END OF DEEPTRACE REPORT
============================================================
"""


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report_text
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print(
    "\n================================================"
)

print(
    "       DEEPTRACE FORENSIC REPORT"
)

print(
    "================================================"
)

print(
    f"\nVideo: {video_name}"
)

print(
    f"Final assessment: "
    f"{final_assessment}"
)

print(
    f"Combined evidence score: "
    f"{combined_score:.4f}"
)

print(
    f"Highest forensic frame: "
    f"{highest_frame_name}"
)

print(
    f"Highest forensic score: "
    f"{highest_frame_data['forensic_evidence_score']:.4f}"
)

print(
    "\nReport generated successfully!"
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)