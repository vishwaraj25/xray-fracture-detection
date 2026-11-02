# Grad-CAM: cnn_resnet18

Pointing game on test fractures the model correctly flagged: how often the Grad-CAM peak lies on an annotated fracture box. The CAM grid is coarse (32 px cells on a 224 px input), so both a strict and a one-cell tolerance are shown.

| images | n | peak within 8 px | peak within one CAM cell (32 px) |
|---|---|---|---|
| all | 92 | 25% | 61% |
| no implant | 77 | 26% | 58% |
| metal implant | 15 | 20% | 73% |

![Grad-CAM examples](figures/gradcam_cnn_resnet18.png)
