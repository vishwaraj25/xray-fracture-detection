"""
Fine-tune an ImageNet-pretrained ResNet-18 (transfer learning) to detect fractures.

Run build_dataset.py first. With --with-graz, clean Graz train/validation patients
are added to training and early stopping (run graz.py --prepare first). Uses the Apple Silicon GPU (MPS) or CUDA when available
(about 80 minutes on an M2).

The best epoch (by validation PR-AUC) is saved to models/cnn.pt and its
validation/test scores to reports/predictions/cnn.csv for evaluate.py.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from config import (CACHE_SIZE, CNN_IMAGE_SIZE, CNN_PATH, IMAGES_DIR, METADATA_PATH,
                    PREDICTIONS_DIR, RAW_DIR, SEED)
from preprocessing import load_square

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Images are letterboxed slightly larger than the input so random crops have room
LOAD_SIZE = round(CNN_IMAGE_SIZE * 8 / 7)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(pretrained=True):
    """ImageNet ResNet-18 with a single-logit head (fracture vs not)."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_model(device):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CNN_PATH, map_location="cpu")["state_dict"])
    return model.to(device).eval()


def train_transform():
    return v2.Compose([
        v2.RandomResizedCrop(CNN_IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return v2.Compose([
        v2.Resize(CNN_IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def to_tensor(gray):
    """uint8 HxW grayscale -> uint8 3xHxW tensor (ImageNet models expect 3 channels)."""
    return torch.from_numpy(gray).unsqueeze(0).expand(3, -1, -1).contiguous()


def image_path(row):
    """Use the downscaled cache when it is big enough, otherwise the original image."""
    if LOAD_SIZE <= CACHE_SIZE:
        return IMAGES_DIR / row["path"]
    return RAW_DIR / "images" / row["folder"] / row["image_id"]


def source_path(row):
    """Graz rows carry their own cached file; FracAtlas rows are found from the metadata."""
    file = row.get("image_file")
    return file if isinstance(file, str) else image_path(row)


class XrayDataset(Dataset):
    """Holds every image in memory so training is not disk-bound."""

    def __init__(self, frame, transform):
        self.images = [to_tensor(load_square(source_path(row), LOAD_SIZE))
                       for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Loading", leave=False)]
        self.labels = torch.tensor(frame["fractured"].to_numpy(), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


@torch.no_grad()
def predict(model, loader, device):
    """Sigmoid scores, averaged with a horizontally flipped copy (test-time augmentation)."""
    model.eval()
    scores = []
    for images, _ in loader:
        images = images.to(device)
        logits = (model(images) + model(torch.flip(images, dims=[3]))) / 2
        scores.append(torch.sigmoid(logits).squeeze(1).cpu())
    return torch.cat(scores).numpy()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on FracAtlas")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--with-graz", action="store_true",
                        help="also train and validate on Graz patients (run graz.py --prepare first)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Training ResNet-18 at {CNN_IMAGE_SIZE} px on {device}")

    df = pd.read_csv(METADATA_PATH)
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if args.with_graz:
        from graz import training_frames

        for split, graz in training_frames().items():
            frames[split] = pd.concat([frames[split], graz], ignore_index=True)
        print(f"With Graz: {len(frames['train'])} training and {len(frames['val'])} validation images")

    train_ds = XrayDataset(frames["train"], train_transform())
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(XrayDataset(frames["val"], eval_transform()), batch_size=32),
        "test": DataLoader(XrayDataset(frames["test"], eval_transform()), batch_size=32),
    }

    model = build_model().to(device)
    # Weight positives by the class ratio so the 18% fractured class is not ignored
    n_pos = frames["train"]["fractured"].sum()
    pos_weight = torch.tensor([(len(frames["train"]) - n_pos) / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loaders["train"]),
        pct_start=0.15,
    )

    CNN_PATH.parent.mkdir(parents=True, exist_ok=True)
    y_val = frames["val"]["fractured"].to_numpy()
    best_ap, best_epoch, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running = 0.0
        for images, labels in tqdm(loaders["train"], desc=f"Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(labels)

        val_scores = predict(model, loaders["val"], device)
        val_ap = average_precision_score(y_val, val_scores)
        val_auc = roc_auc_score(y_val, val_scores)
        train_loss = running / len(train_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_ap, "val_roc_auc": val_auc})
        print(f"epoch {epoch:>2}  loss {train_loss:.4f}  val PR-AUC {val_ap:.3f}  "
              f"ROC-AUC {val_auc:.3f}  ({time.time() - start:.0f}s)", flush=True)

        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            torch.save({"arch": "resnet18", "input_size": CNN_IMAGE_SIZE, "state_dict": model.state_dict()},
                       CNN_PATH)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping: no improvement since epoch {best_epoch}")
            break

    model = load_model(device)
    rows = []
    for split in ("val", "test"):
        scores = predict(model, loaders[split], device)
        # FracAtlas images only; graz.py scores the Graz images
        fracatlas = frames[split]["region"].notna().to_numpy()
        part = frames[split].loc[fracatlas, ["image_id", "split", "region", "hardware"]].copy()
        part["y_true"] = frames[split].loc[fracatlas, "fractured"].to_numpy()
        part["score"] = scores[fracatlas]
        rows.append(part)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(PREDICTIONS_DIR / "cnn.csv", index=False)

    CNN_PATH.with_name("cnn_history.json").write_text(
        json.dumps({"best_epoch": best_epoch, "args": vars(args), "history": history}, indent=2)
    )
    print(f"Best epoch {best_epoch} (val PR-AUC {best_ap:.3f}); saved {CNN_PATH.name}")


if __name__ == "__main__":
    main()
