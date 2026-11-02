"""
Grad-CAM explanations for the CNN, checked against FracAtlas fracture boxes.

  python src/explain.py --arch resnet18

Writes:
  - reports/figures/gradcam_examples.png: heatmaps with the radiologist's boxes
  - a "pointing game" score: for fractured test images the model flags, how
    often the hottest Grad-CAM pixel falls inside an annotated fracture box.
    High = the model looks at the fracture; low = it relies on something else.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import FIGURES_DIR, IMAGES_DIR, METADATA_PATH, MODELS_DIR, PREDICTIONS_DIR, RAW_DIR, REPORTS_DIR
from features import load_square
from plotting import FRACTURED, TEXT_MUTED, plt, save
from train_cnn import EVAL_TRANSFORM, INPUT_SIZE, LOAD_SIZE, build_model, get_device, to_tensor

TARGET_LAYERS = {
    "resnet18": lambda m: m.layer4,
    "resnet50": lambda m: m.layer4,
    "densenet121": lambda m: m.features,
    "efficientnet_b0": lambda m: m.features,
}


# The last conv block of these networks has stride 32
CAM_CELL = 32


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image):
        self.model.zero_grad()
        logit = self.model(image.unsqueeze(0))
        logit.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam / (cam.max() + 1e-8), torch.sigmoid(logit).item()


def fracture_boxes(image_id, cached_shape):
    """YOLO boxes (normalised to the original image) mapped into the square model input."""
    path = RAW_DIR / "Annotations" / "YOLO" / image_id.replace(".jpg", ".txt")
    if not path.exists():
        return []
    h, w = cached_shape
    scale = LOAD_SIZE / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    left, top = (LOAD_SIZE - new_w) // 2, (LOAD_SIZE - new_h) // 2
    to_input = INPUT_SIZE / LOAD_SIZE
    boxes = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        _, cx, cy, bw, bh = map(float, line.split())
        x0 = (left + (cx - bw / 2) * new_w) * to_input
        y0 = (top + (cy - bh / 2) * new_h) * to_input
        boxes.append((x0, y0, bw * new_w * to_input, bh * new_h * to_input))
    return boxes


def hit(cam, boxes, margin=8):
    """Pointing game: is the CAM peak inside (or within `margin` px of) any box?"""
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    return any(bx - margin <= x <= bx + bw + margin and by - margin <= y <= by + bh + margin
               for bx, by, bw, bh in boxes)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for the fracture CNN")
    parser.add_argument("--arch", default="resnet18", choices=TARGET_LAYERS)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    model = build_model(args.arch, pretrained=False)
    state = torch.load(MODELS_DIR / f"cnn_{args.arch}.pt", map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    cam = GradCAM(model, TARGET_LAYERS[args.arch](model))

    meta = pd.read_csv(METADATA_PATH).set_index("image_id")
    preds = pd.read_csv(PREDICTIONS_DIR / f"cnn_{args.arch}.csv")
    val = preds[preds["split"] == "val"]
    # Same rule as evaluate.py: threshold giving 85% sensitivity on validation
    pos_scores = np.sort(val.loc[val["y_true"] == 1, "score"].to_numpy())[::-1]
    threshold = pos_scores[int(np.ceil(0.85 * len(pos_scores))) - 1]

    flagged = preds[(preds["split"] == "test") & (preds["y_true"] == 1) & (preds["score"] >= threshold)]
    results = []
    for image_id in flagged["image_id"]:
        gray = load_square(IMAGES_DIR / meta.loc[image_id, "path"], LOAD_SIZE)
        tensor = EVAL_TRANSFORM(to_tensor(gray)).to(device)
        heatmap, score = cam(tensor)
        cached = plt.imread(IMAGES_DIR / meta.loc[image_id, "path"]).shape[:2]
        boxes = fracture_boxes(image_id, cached)
        if boxes:
            results.append({"image_id": image_id, "score": score, "hit": hit(heatmap, boxes),
                            "hit_cell": hit(heatmap, boxes, margin=CAM_CELL),
                            "hardware": meta.loc[image_id, "hardware"],
                            "dup_group": meta.loc[image_id, "dup_group"],
                            "heatmap": heatmap, "boxes": boxes,
                            "image": np.asarray(EVAL_TRANSFORM.transforms[0](to_tensor(gray))[0])})

    table = pd.DataFrame(results)
    summary = {
        "all": table,
        "no implant": table[table["hardware"] == 0],
        "metal implant": table[table["hardware"] == 1],
    }
    lines = ["| images | n | peak within 8 px | peak within one CAM cell (32 px) |", "|---|---|---|---|"]
    for label, part in summary.items():
        lines.append(f"| {label} | {len(part)} | {part['hit'].mean():.0%} | {part['hit_cell'].mean():.0%} |")
    print("\n".join(lines))

    # Show the most confident hits and misses, one image per near-duplicate group
    unique = table.drop_duplicates("dup_group").to_dict("records")
    hits_list = sorted([r for r in unique if r["hit_cell"]], key=lambda r: -r["score"])
    miss_list = sorted([r for r in unique if not r["hit_cell"]], key=lambda r: -r["score"])
    half = args.examples // 2
    chosen = hits_list[:half] + miss_list[:args.examples - min(half, len(hits_list))]
    chosen = chosen[:args.examples]
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
    save(fig, FIGURES_DIR / "gradcam_examples.png")

    (REPORTS_DIR / f"gradcam_{args.arch}.md").write_text(
        f"# Grad-CAM: {args.arch}\n\n"
        "Pointing game on test fractures the model correctly flagged: how often the Grad-CAM peak "
        f"lies on an annotated fracture box. The CAM grid is coarse ({CAM_CELL} px cells on a "
        f"{INPUT_SIZE} px input), so both a strict and a one-cell tolerance are shown.\n\n"
        + "\n".join(lines) + "\n\n"
        "![Grad-CAM examples](figures/gradcam_examples.png)\n"
    )


if __name__ == "__main__":
    main()
