from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class DeepTraceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            label_dir = self.root_dir / label_name

            for file_path in label_dir.rglob("*.pt"):
                self.samples.append((file_path, label))

        print(f"Dataset: {self.root_dir}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]

        image = torch.load(file_path)

        return image, torch.tensor(label, dtype=torch.float32)


def create_dataloader(root_dir, batch_size=16, shuffle=True):
    dataset = DeepTraceDataset(root_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


if __name__ == "__main__":
    train_dir = Path(__file__).resolve().parent.parent / "dataset" / "normalized" / "train"

    loader = create_dataloader(train_dir)

    images, labels = next(iter(loader))

    print("DataLoader test successful!")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)