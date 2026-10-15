import gzip
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
TRAIN_IMAGES = ROOT / "data" / "train-images-idx3-ubyte.gz"
TRAIN_LABELS = ROOT / "data" / "train-labels-idx1-ubyte.gz"
TEST_IMAGES = ROOT / "x_test_a87ff6.npy"
OUTPUT = ROOT / "y_test.npy"
SEED = 42
EPOCHS = 15


def read_idx_images(path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file header in {path}")
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(count, rows, columns)


def read_idx_labels(path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file header in {path}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
        if len(labels) != count:
            raise ValueError(f"Unexpected label count in {path}")
        return labels


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    images = read_idx_images(TRAIN_IMAGES)
    labels = read_idx_labels(TRAIN_LABELS).astype(np.int64)
    if len(images) != len(labels):
        raise ValueError("Training image and label counts do not match")

    images = torch.from_numpy(images.copy()).float().div_(255).unsqueeze(1)
    mean, std = images.mean(), images.std()
    images = (images - mean) / std
    labels = torch.from_numpy(labels)

    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(len(labels), generator=generator)
    val_indices, train_indices = permutation[:5000], permutation[5000:]
    train_loader = DataLoader(
        TensorDataset(images[train_indices], labels[train_indices]),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    val_images, val_labels = images[val_indices].to(device), labels[val_indices].to(device)

    model = FashionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_accuracy = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_images), batch_labels, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_labels)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            correct = sum(
                (model(batch).argmax(1) == target).sum().item()
                for batch, target in zip(
                    val_images.split(512), val_labels.split(512)
                )
            )
        accuracy = correct / len(val_labels)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS}: "
            f"loss={loss_sum / len(train_indices):.4f}, val_accuracy={accuracy:.4%}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()

    test_images = np.load(TEST_IMAGES)
    if test_images.ndim != 3 or test_images.shape[1:] != (28, 28):
        raise ValueError(f"Unexpected test image shape: {test_images.shape}")
    test_tensor = torch.from_numpy(test_images.copy()).float().div_(255).unsqueeze(1)
    test_tensor = (test_tensor - mean.cpu()) / std.cpu()
    predictions = []
    with torch.no_grad():
        for batch in test_tensor.split(512):
            predictions.extend(model(batch.to(device)).argmax(1).cpu().tolist())

    result = np.asarray(predictions, dtype=np.int64)
    if result.shape != (len(test_images),) or not np.all((0 <= result) & (result < 10)):
        raise RuntimeError("Generated predictions have an invalid shape or class index")
    np.save(OUTPUT, result)
    print(f"Saved {len(result)} predictions to {OUTPUT}")
    print(f"Validation accuracy: {best_accuracy:.4%}")


if __name__ == "__main__":
    main()
