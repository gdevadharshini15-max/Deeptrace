import sys
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_step(name, command):
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)

    result = subprocess.run(
        command,
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"\n❌ {name} failed.")
        sys.exit(result.returncode)

    print(f"[OK] {name} completed successfully.")


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python run_deeptrace.py <video_path>")
        print("\nExample:")
        print("python run_deeptrace.py dataset/raw/real/v3.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    video_path = video_path.resolve()

    print("\n" + "=" * 70)
    print("        DEEPTRACE - FORENSIC DEEPFAKE ANALYSIS")
    print("=" * 70)
    print(f"Video: {video_path}")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Video prediction + suspicious frame extraction
    # ---------------------------------------------------------
    run_step(
        "Video Prediction",
        [
            sys.executable,
            "evaluation/predict_video.py",
            str(video_path)
        ]
    )

    # ---------------------------------------------------------
    # 2. Temporal analysis
    # ---------------------------------------------------------
    run_step(
        "Temporal Analysis",
        [
            sys.executable,
            "analysis/temporal_analysis.py"
        ]
    )

    # ---------------------------------------------------------
    # 3. Grad-CAM explainability
    # ---------------------------------------------------------
    run_step(
        "Grad-CAM Explainability",
        [
            sys.executable,
            "explainability/gradcam.py"
        ]
    )

    # ---------------------------------------------------------
    # 4. Evidence fusion
    # ---------------------------------------------------------
    run_step(
        "Forensic Evidence Fusion",
        [
            sys.executable,
            "analysis/evidence_fusion.py"
        ]
    )

    # ---------------------------------------------------------
    # 5. Automatic forensic report
    # ---------------------------------------------------------
    run_step(
        "Automatic Report Generation",
        [
            sys.executable,
            "analysis/generate_report.py"
        ]
    )

    print("\n" + "=" * 70)
    print("DEEPTRACE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print("\nGenerated results:")
    print("Suspicious frames  -> results/suspicious_frames/")
    print("Grad-CAM           -> results/explainability/")
    print("Temporal analysis  -> results/temporal_analysis/")
    print("Forensic fusion    -> results/forensic_fusion/")
    print("Final report       -> results/reports/")
    print("=" * 70)


if __name__ == "__main__":
    main()