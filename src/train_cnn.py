"""
Fine-tune an ImageNet-pretrained CNN (transfer learning) to detect fractures.

Run prepare_data.py first. Uses Apple Silicon GPU (MPS) or CUDA when available.

  python src/train_cnn.py --arch resnet18 --epochs 25
  python src/train_cnn.py --arch resnet18 --size 384      # higher resolution

The best epoch (by validation PR-AUC) is saved to models/<run>.pt and its
validation/test scores to reports/predictions/<run>.csv for evaluate.py, where
<run> is cnn_<arch> at 224 px and cnn_<arch>_<size> otherwise.
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

from config import CACHE_SIZE, IMAGES_DIR, METADATA_PATH, MODELS_DIR, PREDICTIONS_DIR, RAW_DIR, SEED
from features import load_square

DEFAULT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT, "classifier"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier"),
}


def run_name(arch, size):
    return f"cnn_{arch}" if size == DEFAULT_SIZE else f"cnn_{arch}_{size}"


def load_size(input_size):
    """Images are letterboxed slightly larger than the input so random crops have room."""
    return round(input_size * 8 / 7)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(arch, pretrained=True):
    """ImageNet backbone with a single-logit head."""
    constructor, weights, head = ARCHITECTURES[arch]
    model = constructor(weights=weights if pretrained else None)
    if head == "fc":
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        last = model.classifier[-1] if isinstance(model.classifier, nn.Sequential) else model.classifier
        new_head = nn.Linear(last.in_features, 1)
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = new_head
        else:
            model.classifier = new_head
    return model


def train_transform(input_size):
    return v2.Compose([
        v2.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(input_size):
    return v2.Compose([
        v2.Resize(input_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def to_tensor(gray):
    """uint8 HxW grayscale -> uint8 3xHxW tensor (ImageNet models expect 3 channels)."""
    return torch.from_numpy(gray).unsqueeze(0).expand(3, -1, -1).contiguous()


def image_path(row, size):
    """Use the downscaled cache when it is big enough, otherwise the original image."""
    if size <= CACHE_SIZE:
        return IMAGES_DIR / row["path"]
    return RAW_DIR / "images" / row["folder"] / row["image_id"]


class XrayDataset(Dataset):
    """Holds every image in memory so training is not disk-bound."""

    def __init__(self, frame, input_size, transform):
        size = load_size(input_size)
        self.images = [to_tensor(load_square(image_path(row, size), size))
                       for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Loading", leave=False)]
        self.labels = torch.tensor(frame["fractured"].to_numpy(), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


def load_checkpoint(name, device):
    """Model from models/<name>.pt plus its input size (older checkpoints are 224 px)."""
    checkpoint = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu")
    model = build_model(checkpoint["arch"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["arch"], checkpoint.get("input_size", DEFAULT_SIZE)


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
    parser = argparse.ArgumentParser(description="Fine-tune a CNN on FracAtlas")
    parser.add_argument("--arch", default="resnet18", choices=ARCHITECTURES)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="input resolution in pixels")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    name = run_name(args.arch, args.size)
    print(f"Run {name} on {device}")

    df = pd.read_csv(METADATA_PATH)
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}

    train_ds = XrayDataset(frames["train"], args.size, train_transform(args.size))
    evaluate_tf = eval_transform(args.size)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(XrayDataset(frames["val"], args.size, evaluate_tf), batch_size=32),
        "test": DataLoader(XrayDataset(frames["test"], args.size, evaluate_tf), batch_size=32),
    }

    model = build_model(args.arch).to(device)
    # Weight positives by the class ratio so the 18% fractured class is not ignored
    n_pos = frames["train"]["fractured"].sum()
    pos_weight = torch.tensor([(len(frames["train"]) - n_pos) / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loaders["train"]),
        pct_start=0.15,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODELS_DIR / f"{name}.pt"
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
            torch.save({"arch": args.arch, "input_size": args.size, "state_dict": model.state_dict()},
                       checkpoint)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping: no improvement since epoch {best_epoch}")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
    rows = []
    for split in ("val", "test"):
        part = frames[split][["image_id", "split", "region", "hardware"]].copy()
        part["y_true"] = frames[split]["fractured"].to_numpy()
        part["score"] = predict(model, loaders[split], device)
        rows.append(part)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(PREDICTIONS_DIR / f"{name}.csv", index=False)

    (MODELS_DIR / f"{name}_history.json").write_text(
        json.dumps({"best_epoch": best_epoch, "args": vars(args), "history": history}, indent=2)
    )
    print(f"Best epoch {best_epoch} (val PR-AUC {best_ap:.3f}); saved {checkpoint.name}")


if __name__ == "__main__":
    main()
