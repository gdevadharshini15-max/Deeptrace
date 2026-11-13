import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "results" / "suspicious_frames"
OUTPUT_DIR = BASE_DIR / "results" / "forensic_frequency"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_frequency(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image to floating point
    gray_float = np.float32(gray)

    # Apply 2D Fast Fourier Transform
    fft = np.fft.fft2(gray_float)

    # Shift low frequencies to the center
    fft_shift = np.fft.fftshift(fft)

    # Calculate magnitude spectrum
    magnitude = np.abs(fft_shift)

    # Log scale for visualization
    magnitude_spectrum = np.log1p(magnitude)

    # Calculate frequency statistics
    mean_frequency = np.mean(magnitude_spectrum)
    std_frequency = np.std(magnitude_spectrum)
    max_frequency = np.max(magnitude_spectrum)

    # Normalize for visualization
    if max_frequency > 0:
        visual = (
            magnitude_spectrum / max_frequency * 255
        ).astype(np.uint8)
    else:
        visual = np.zeros_like(gray, dtype=np.uint8)

    # Apply color map
    frequency_visual = cv2.applyColorMap(
        visual,
        cv2.COLORMAP_JET
    )

    # Add information
    text1 = f"Mean Frequency: {mean_frequency:.2f}"
    text2 = f"Frequency Std: {std_frequency:.2f}"

    cv2.putText(
        frequency_visual,
        text1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frequency_visual,
        text2,
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    output_path = (
        OUTPUT_DIR /
        f"frequency_{image_path.name}"
    )

    cv2.imwrite(str(output_path), frequency_visual)

    print(f"\nFrame: {image_path.name}")
    print(f"Mean frequency magnitude: {mean_frequency:.4f}")
    print(f"Frequency variation: {std_frequency:.4f}")
    print(f"Maximum frequency magnitude: {max_frequency:.4f}")
    print(f"Saved: {output_path}")


image_files = list(INPUT_DIR.glob("*.jpg"))

if len(image_files) == 0:
    print("No suspicious frames found.")
else:

    print("===== FREQUENCY-DOMAIN FORENSIC ANALYSIS =====")

    for image_path in image_files:
        analyze_frequency(image_path)

print("\nFrequency analysis completed!")