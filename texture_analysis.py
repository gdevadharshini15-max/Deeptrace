import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "results" / "suspicious_frames"
OUTPUT_DIR = BASE_DIR / "results" / "forensic_texture"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_texture(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate local texture variation using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Texture sharpness / variation score
    texture_score = laplacian.var()

    # Create a visual representation
    laplacian_abs = np.absolute(laplacian)

    # Normalize for visualization
    if laplacian_abs.max() > 0:
        laplacian_visual = (
            laplacian_abs / laplacian_abs.max() * 255
        ).astype(np.uint8)
    else:
        laplacian_visual = np.zeros_like(gray)

    texture_visual = cv2.applyColorMap(
        laplacian_visual,
        cv2.COLORMAP_JET
    )

    # Add forensic information
    text = f"Texture Variation: {texture_score:.2f}"

    cv2.putText(
        texture_visual,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    output_path = (
        OUTPUT_DIR /
        f"texture_{image_path.name}"
    )

    cv2.imwrite(str(output_path), texture_visual)

    print(f"\nFrame: {image_path.name}")
    print(f"Texture variation score: {texture_score:.2f}")
    print(f"Saved: {output_path}")


image_files = list(INPUT_DIR.glob("*.jpg"))

if len(image_files) == 0:
    print("No suspicious frames found.")
else:

    print("===== TEXTURE FORENSIC ANALYSIS =====")

    for image_path in image_files:
        analyze_texture(image_path)

print("\nTexture analysis completed!")