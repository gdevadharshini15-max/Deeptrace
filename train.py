from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.cnn_model import DeepTraceCNN
from training.dataset_loader import create_dataloader


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "normalized" / "train"
VAL_DIR = BASE_DIR / "dataset" / "normalized" / "val"

# Training settings
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# DataLoaders
train_loader = create_dataloader(
    TRAIN_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = create_dataloader(
    VAL_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Model
model = DeepTraceCNN().to(DEVICE)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Training
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = (torch.sigmoid(outputs) >= 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs = model(images)

            predictions = (torch.sigmoid(outputs) >= 0.5).float()

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


# Save model
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "deeptrace_cnn.pth"

torch.save(model.state_dict(), model_path)

print("\nTraining completed!")
print("Model saved to:", model_path)