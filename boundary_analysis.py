import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "results" / "suspicious_frames"
OUTPUT_DIR = BASE_DIR / "results" / "forensic_boundary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_boundary(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Calculate edge density
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]

    edge_density = edge_pixels / total_pixels

    # Create visualization
    edge_visual = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Add forensic information
    text = f"Boundary Edge Density: {edge_density:.4f}"

    cv2.putText(
        edge_visual,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    output_path = OUTPUT_DIR / f"boundary_{image_path.name}"

    cv2.imwrite(str(output_path), edge_visual)

    print(f"\nFrame: {image_path.name}")
    print(f"Edge pixels: {edge_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Edge density: {edge_density:.4f}")
    print(f"Saved: {output_path}")


image_files = list(INPUT_DIR.glob("*.jpg"))

if len(image_files) == 0:
    print("No suspicious frames found.")
else:
    print("===== FACE-BOUNDARY FORENSIC ANALYSIS =====")

    for image_path in image_files:
        analyze_boundary(image_path)

print("\nBoundary analysis completed!")