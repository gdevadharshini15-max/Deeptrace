import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "results" / "suspicious_frames"
OUTPUT_DIR = BASE_DIR / "results" / "forensic_color"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_color(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        return

    # OpenCV reads images as BGR
    b, g, r = cv2.split(image)

    # Convert to HSV for illumination analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Calculate mean color values
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    # Calculate brightness statistics
    mean_brightness = np.mean(v)
    brightness_std = np.std(v)

    # Calculate saturation
    mean_saturation = np.mean(s)

    # Color imbalance
    color_imbalance = (
        abs(mean_r - mean_g)
        + abs(mean_g - mean_b)
        + abs(mean_r - mean_b)
    )

    # Create brightness visualization
    brightness_visual = cv2.applyColorMap(
        v,
        cv2.COLORMAP_JET
    )

    text1 = f"Brightness: {mean_brightness:.2f}"
    text2 = f"Brightness Std: {brightness_std:.2f}"

    cv2.putText(
        brightness_visual,
        text1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        brightness_visual,
        text2,
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    output_path = (
        OUTPUT_DIR /
        f"color_{image_path.name}"
    )

    cv2.imwrite(str(output_path), brightness_visual)

    print(f"\nFrame: {image_path.name}")
    print(f"Mean B: {mean_b:.2f}")
    print(f"Mean G: {mean_g:.2f}")
    print(f"Mean R: {mean_r:.2f}")
    print(f"Mean brightness: {mean_brightness:.2f}")
    print(f"Brightness variation: {brightness_std:.2f}")
    print(f"Mean saturation: {mean_saturation:.2f}")
    print(f"Color imbalance: {color_imbalance:.2f}")
    print(f"Saved: {output_path}")


image_files = list(INPUT_DIR.glob("*.jpg"))

if len(image_files) == 0:
    print("No suspicious frames found.")
else:

    print("===== COLOR / ILLUMINATION FORENSIC ANALYSIS =====")

    for image_path in image_files:
        analyze_color(image_path)

print("\nColor analysis completed!")