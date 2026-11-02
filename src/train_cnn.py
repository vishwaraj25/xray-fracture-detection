"""
Fine-tune an ImageNet-pretrained CNN (transfer learning) to detect fractures.

Run prepare_data.py first. Uses Apple Silicon GPU (MPS) or CUDA when available.

  python src/train_cnn.py --arch resnet18 --epochs 25

The best epoch (by validation PR-AUC) is saved to models/cnn_<arch>.pt and its
validation/test scores to reports/predictions/cnn_<arch>.csv for evaluate.py.
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

from config import IMAGES_DIR, METADATA_PATH, MODELS_DIR, PREDICTIONS_DIR, SEED
from features import load_square

LOAD_SIZE = 256
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT, "classifier"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier"),
}


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


TRAIN_TRANSFORM = v2.Compose([
    v2.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = v2.Compose([
    v2.Resize(INPUT_SIZE, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def to_tensor(gray):
    """uint8 HxW grayscale -> uint8 3xHxW tensor (ImageNet models expect 3 channels)."""
    return torch.from_numpy(gray).unsqueeze(0).expand(3, -1, -1).contiguous()


class XrayDataset(Dataset):
    """Holds every image in memory (~250 MB) so training is not disk-bound."""

    def __init__(self, frame, transform):
        self.images = [to_tensor(load_square(IMAGES_DIR / p, LOAD_SIZE)) for p in frame["path"]]
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
    parser = argparse.ArgumentParser(description="Fine-tune a CNN on FracAtlas")
    parser.add_argument("--arch", default="resnet18", choices=ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print("Device:", device)

    df = pd.read_csv(METADATA_PATH)
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}

    train_ds = XrayDataset(frames["train"], TRAIN_TRANSFORM)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(XrayDataset(frames["val"], EVAL_TRANSFORM), batch_size=64),
        "test": DataLoader(XrayDataset(frames["test"], EVAL_TRANSFORM), batch_size=64),
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
    checkpoint = MODELS_DIR / f"cnn_{args.arch}.pt"
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
              f"ROC-AUC {val_auc:.3f}  ({time.time() - start:.0f}s)")

        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            torch.save({"arch": args.arch, "state_dict": model.state_dict()}, checkpoint)
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
    pd.concat(rows).to_csv(PREDICTIONS_DIR / f"cnn_{args.arch}.csv", index=False)

    (MODELS_DIR / f"cnn_{args.arch}_history.json").write_text(
        json.dumps({"best_epoch": best_epoch, "args": vars(args), "history": history}, indent=2)
    )
    print(f"Best epoch {best_epoch} (val PR-AUC {best_ap:.3f}); saved {checkpoint.name}")


if __name__ == "__main__":
    main()
