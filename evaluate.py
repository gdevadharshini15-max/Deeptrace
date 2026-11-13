import sys
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.cnn_model import DeepTraceCNN
from training.dataset_loader import create_dataloader


# ==================================================
# PATHS
# ==================================================

TEST_DIR = BASE_DIR / "dataset" / "normalized" / "test"
MODEL_PATH = BASE_DIR / "models" / "deeptrace_cnn.pth"

RESULTS_DIR = BASE_DIR / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RESULTS_DIR / "test_metrics.txt"
CONFUSION_FILE = RESULTS_DIR / "confusion_matrix.png"
PREDICTIONS_FILE = RESULTS_DIR / "predictions.csv"


# ==================================================
# DEVICE
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("Test dataset:", TEST_DIR)
print("Model:", MODEL_PATH)


# ==================================================
# LOAD DATASET
# ==================================================

test_loader = create_dataloader(
    TEST_DIR,
    batch_size=16,
    shuffle=False
)


# ==================================================
# LOAD MODEL
# ==================================================

model = DeepTraceCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")


# ==================================================
# PREDICTION
# ==================================================

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
        ).squeeze(1)

        predictions = (
            probabilities >= 0.5
        ).int()

        all_labels.extend(
            labels.int().cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


# ==================================================
# CONFUSION MATRIX VALUES
# ==================================================

tp = np.sum(
    (all_labels == 1) &
    (all_predictions == 1)
)

tn = np.sum(
    (all_labels == 0) &
    (all_predictions == 0)
)

fp = np.sum(
    (all_labels == 0) &
    (all_predictions == 1)
)

fn = np.sum(
    (all_labels == 1) &
    (all_predictions == 0)
)


# ==================================================
# METRICS
# ==================================================

total = len(all_labels)

accuracy = (
    (tp + tn) / total
    if total > 0
    else 0
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ==================================================
# PRINT RESULTS
# ==================================================

print("\n==========================================")
print("       DEEPTRACE FINAL EVALUATION")
print("==========================================")

print(f"\nTest samples : {total}")

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")

print("\nConfusion Matrix:")
print(
    np.array([
        [tn, fp],
        [fn, tp]
    ])
)

print("\nClass labels:")
print("0 = REAL")
print("1 = FAKE")


# ==================================================
# SAVE METRICS
# ==================================================

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DEEPTRACE FINAL MODEL EVALUATION\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        f"Test samples : {total}\n"
    )

    file.write(
        f"Accuracy     : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision    : {precision:.4f}\n"
    )

    file.write(
        f"Recall       : {recall:.4f}\n"
    )

    file.write(
        f"F1 Score     : {f1:.4f}\n"
    )

    file.write("\nConfusion Matrix:\n")

    file.write(
        f"[[{tn}, {fp}],\n"
        f" [{fn}, {tp}]]\n"
    )

    file.write(
        "\nClass labels:\n"
    )

    file.write(
        "0 = REAL\n"
    )

    file.write(
        "1 = FAKE\n"
    )


# ==================================================
# SAVE PREDICTIONS
# ==================================================

with open(
    PREDICTIONS_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "sample_index",
        "actual_label",
        "predicted_label",
        "fake_probability"
    ])

    for i in range(total):

        writer.writerow([
            i,
            int(all_labels[i]),
            int(all_predictions[i]),
            float(all_probabilities[i])
        ])


# ==================================================
# CONFUSION MATRIX VISUALIZATION
# ==================================================

cm = np.array([
    [tn, fp],
    [fn, tp]
])

fig, ax = plt.subplots(
    figsize=(6, 5)
)

image = ax.imshow(cm)

ax.set_title(
    "DeepTrace Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "REAL",
    "FAKE"
])

ax.set_yticklabels([
    "REAL",
    "FAKE"
])


for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_FILE,
    dpi=300
)

plt.close()


# ==================================================
# FINISH
# ==================================================

print("\nFiles generated successfully!")

print("\nMetrics:")
print(METRICS_FILE)

print("\nConfusion matrix:")
print(CONFUSION_FILE)

print("\nPredictions:")
print(PREDICTIONS_FILE)

print("\n==========================================")
print("          EVALUATION COMPLETED")
print("==========================================")
