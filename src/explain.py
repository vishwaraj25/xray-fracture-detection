"""
Grad-CAM explanations for the CNN, checked against the FracAtlas fracture boxes.

Writes:
  - reports/predictions/gradcam.csv: the pointing-game result for each image
  - reports/figures/gradcam.png: heatmaps with the radiologists' boxes
  - reports/gradcam.md: a "pointing game" score. For fractured test images the
    CNN flags, how often the hottest Grad-CAM pixel falls on an annotated fracture
    box. High = the model looks at the fracture; low = it relies on something else.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import CNN_IMAGE_SIZE, FIGURES_DIR, METADATA_PATH, PREDICTIONS_DIR, RAW_DIR, REPORTS_DIR
from evaluate import validation_threshold
from plotting import FRACTURED, TEXT_MUTED, plt, save
from preprocessing import load_square
from train_cnn import LOAD_SIZE, eval_transform, get_device, image_path, load_model, to_tensor

# ResNet's last conv block has stride 32, so each heatmap cell covers 32 px
CAM_CELL = 32
# Strict pointing-game tolerance in pixels (8 px on a 224 px input, scaled up)
STRICT_MARGIN = 8 * CNN_IMAGE_SIZE / 224


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        model.layer4.register_forward_hook(self._save_activation)
        model.layer4.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image):
        """Heatmap at the input's resolution (normalised to 0..1) and the sigmoid score."""
        self.model.zero_grad()
        logit = self.model(image.unsqueeze(0))
        logit.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(image.shape[-2:]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam / (cam.max() + 1e-8), torch.sigmoid(logit).item()


def annotated_boxes(image_id):
    """Radiologists' fracture boxes as (x, y, w, h), normalised to the original image."""
    path = RAW_DIR / "Annotations" / "YOLO" / image_id.replace(".jpg", ".txt")
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append((cx - bw / 2, cy - bh / 2, bw, bh))
    return boxes


def fracture_boxes(image_id, original_shape):
    """Annotated boxes mapped into the square model input, in pixels."""
    h, w = original_shape
    scale = LOAD_SIZE / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    left, top = (LOAD_SIZE - new_w) // 2, (LOAD_SIZE - new_h) // 2
    to_input = CNN_IMAGE_SIZE / LOAD_SIZE
    return [((left + x * new_w) * to_input, (top + y * new_h) * to_input, bw * new_w * to_input, bh * new_h * to_input)
            for x, y, bw, bh in annotated_boxes(image_id)]


def hit(cam, boxes, margin):
    """Pointing game: is the CAM peak inside (or within `margin` px of) any box?"""
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    return any(bx - margin <= x <= bx + bw + margin and by - margin <= y <= by + bh + margin
               for bx, by, bw, bh in boxes)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for the fracture CNN")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    model = load_model(device)
    transform = eval_transform()
    cam = GradCAM(model)

    meta = pd.read_csv(METADATA_PATH).set_index("image_id")
    preds = pd.read_csv(PREDICTIONS_DIR / "cnn.csv")
    threshold = validation_threshold(preds)

    flagged = preds[(preds["split"] == "test") & (preds["y_true"] == 1) & (preds["score"] >= threshold)]
    results = []
    for image_id in flagged["image_id"]:
        row = meta.loc[image_id]
        gray = load_square(image_path({**row, "image_id": image_id}), LOAD_SIZE)
        tensor = transform(to_tensor(gray)).to(device)
        heatmap, score = cam(tensor)
        boxes = fracture_boxes(image_id, (row["orig_height"], row["orig_width"]))
        if boxes:
            results.append({"image_id": image_id, "score": score, "hit": hit(heatmap, boxes, STRICT_MARGIN),
                            "hit_cell": hit(heatmap, boxes, CAM_CELL),
                            "hardware": row["hardware"], "dup_group": row["dup_group"],
                            "heatmap": heatmap, "boxes": boxes,
                            "image": np.asarray(transform.transforms[0](to_tensor(gray))[0])})

    table = pd.DataFrame(results)
    # Per-image result, used by the web app to pick example images
    table[["image_id", "score", "hit", "hit_cell"]].to_csv(PREDICTIONS_DIR / "gradcam.csv", index=False)
    summary = {
        "all": table,
        "no implant": table[table["hardware"] == 0],
        "metal implant": table[table["hardware"] == 1],
    }
    lines = [f"| images | n | peak within {STRICT_MARGIN:.0f} px | peak within one CAM cell ({CAM_CELL} px) |",
             "|---|---|---|---|"]
    for label, part in summary.items():
        lines.append(f"| {label} | {len(part)} | {part['hit'].mean():.0%} | {part['hit_cell'].mean():.0%} |")
    print("\n".join(lines))

    # Show the most confident hits and misses, one image per near-duplicate group
    unique = table.drop_duplicates("dup_group").to_dict("records")
    hits_list = sorted([r for r in unique if r["hit_cell"]], key=lambda r: -r["score"])
    miss_list = sorted([r for r in unique if not r["hit_cell"]], key=lambda r: -r["score"])
    half = args.examples // 2
    chosen = (hits_list[:half] + miss_list[:args.examples - min(half, len(hits_list))])[:args.examples]
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax, r in zip(np.atleast_1d(axes).ravel(), chosen):
        ax.imshow(r["image"], cmap="gray")
        ax.imshow(r["heatmap"], cmap="inferno", alpha=0.45)
        for bx, by, bw, bh in r["boxes"]:
            ax.add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor="#1baf7a", linewidth=2))
        verdict = "on fracture" if r["hit_cell"] else "elsewhere"
        implant = " (implant)" if r["hardware"] else ""
        ax.set_title(f"p={r['score']:.2f}{implant}\nfocus {verdict}", fontsize=9, loc="center",
                     color=FRACTURED if not r["hit_cell"] else TEXT_MUTED)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    fig.suptitle("Grad-CAM (heat) vs annotated fracture box (green)", x=0.02, ha="left",
                 fontsize=12, fontweight="semibold")
    save(fig, FIGURES_DIR / "gradcam.png")

    (REPORTS_DIR / "gradcam.md").write_text(
        "# Grad-CAM: where the CNN looks\n\n"
        "Pointing game on test fractures the CNN correctly flagged: how often the Grad-CAM peak "
        f"lies on an annotated fracture box. The CAM grid is coarse ({CAM_CELL} px cells on a "
        f"{CNN_IMAGE_SIZE} px input), so both a strict and a one-cell tolerance are shown.\n\n"
        + "\n".join(lines) + "\n\n"
        "![Grad-CAM examples](figures/gradcam.png)\n"
    )


if __name__ == "__main__":
    main()
