import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import torch
import numpy as np

from models.cnn_model import DeepTraceCNN


# ==========================================
# PATHS
# ==========================================

MODEL_PATH = BASE_DIR / "models" / "deeptrace_cnn_final.pth"

INPUT_DIR = BASE_DIR / "results" / "suspicious_frames"

OUTPUT_DIR = BASE_DIR / "results" / "explainability"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# LOAD MODEL
# ==========================================

model = DeepTraceCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")


# ==========================================
# SELECT FINAL CONVOLUTIONAL LAYER
# ==========================================

target_layer = model.features[-2]


# ==========================================
# STORAGE FOR ACTIVATIONS AND GRADIENTS
# ==========================================

activations = None
gradients = None


def forward_hook(module, input, output):

    global activations

    activations = output


def backward_hook(module, grad_input, grad_output):

    global gradients

    gradients = grad_output[0]


# Register hooks
forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ==========================================
# PROCESS IMAGE
# ==========================================

image_files = list(
    INPUT_DIR.glob("*.jpg")
)

if len(image_files) == 0:

    print(
        "No suspicious frames found."
    )

    sys.exit()


for image_path in image_files:

    print(
        "\nProcessing:",
        image_path.name
    )

    # --------------------------------------
    # Read image
    # --------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "Could not read image."
        )

        continue


    # --------------------------------------
    # Resize
    # --------------------------------------

    image = cv2.resize(
        image,
        (224, 224)
    )


    # --------------------------------------
    # BGR -> RGB
    # --------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # Normalize
    # --------------------------------------

    normalized = (
        rgb_image.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------
    # HWC -> CHW
    # --------------------------------------

    tensor = np.transpose(
        normalized,
        (2, 0, 1)
    )


    # --------------------------------------
    # NumPy -> Tensor
    # --------------------------------------

    tensor = torch.from_numpy(
        tensor
    ).float()


    # Add batch dimension
    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)


    # Enable gradient calculation
    tensor.requires_grad_(True)


    # ======================================
    # FORWARD PASS
    # ======================================

    model.zero_grad()

    output = model(
        tensor
    )

    probability = torch.sigmoid(
        output
    ).item()

    prediction = (
        "DEEPFAKE"
        if probability >= 0.5
        else "REAL"
    )


    # ======================================
    # BACKWARD PASS
    # ======================================

    # Use the model output for gradient
    output.backward()


    # ======================================
    # GET ACTIVATIONS + GRADIENTS
    # ======================================

    if activations is None or gradients is None:

        print(
            "Could not obtain Grad-CAM data."
        )

        continue


    activation = activations.detach()

    gradient = gradients.detach()


    # ======================================
    # GLOBAL AVERAGE POOLING OF GRADIENTS
    # ======================================

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )


    # ======================================
    # WEIGHTED ACTIVATION MAP
    # ======================================

    cam = (
        weights * activation
    ).sum(
        dim=1
    ).squeeze()


    # ======================================
    # REMOVE NEGATIVE VALUES
    # ======================================

    cam = torch.relu(
        cam
    )


    # ======================================
    # NORMALIZE HEATMAP
    # ======================================

    cam = cam.cpu().numpy()

    if cam.max() > 0:

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )


    # ======================================
    # RESIZE HEATMAP
    # ======================================

    cam = cv2.resize(
        cam,
        (224, 224)
    )


    # Convert to 0-255
    heatmap = np.uint8(
        255 * cam
    )


    # ======================================
    # CREATE COLOR HEATMAP
    # ======================================

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # ======================================
    # CREATE OVERLAY
    # ======================================

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )


    # ======================================
    # ADD PREDICTION TEXT
    # ======================================

    text = (
        f"{prediction} | "
        f"Fake probability: "
        f"{probability:.2f}"
    )

    cv2.putText(
        overlay,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ======================================
    # SAVE RESULT
    # ======================================

    output_path = (
        OUTPUT_DIR
        / f"gradcam_{image_path.name}"
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )


    print(
        "Prediction:",
        prediction
    )

    print(
        f"Fake probability: "
        f"{probability:.4f}"
    )

    print(
        "Saved:",
        output_path
    )


# ==========================================
# REMOVE HOOKS
# ==========================================

forward_handle.remove()

backward_handle.remove()


print(
    "\nGrad-CAM generation completed!"
)