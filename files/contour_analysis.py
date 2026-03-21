"""
contour_analysis.py
After the SVM predicts FRACTURED, this module highlights suspicious regions
on the original X-ray using filtered contour analysis.

Filtering logic (so output isn't noisy):
  1. Area threshold    — ignore tiny specks (< 80px²)
  2. Circularity       — fracture lines are jagged, not smooth circles
                         circularity = 4π·area / perimeter²
                         keep contours with circularity < 0.45
  3. Top-N by area     — only draw the 5 largest suspicious regions

Returns the original image with coloured contour overlays drawn on it.
"""

import cv2
import numpy as np


def get_suspicious_contours(edges: np.ndarray,
                             min_area: int = 80,
                             max_circularity: float = 0.45,
                             top_n: int = 5) -> list:
    """
    Return a filtered list of contours that look like fracture regions.
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < max_circularity:
            candidates.append((area, cnt))

    # Sort by area descending, take top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [cnt for _, cnt in candidates[:top_n]]


def draw_contour_overlay(original_bgr: np.ndarray,
                         edges: np.ndarray) -> np.ndarray:
    """
    Draw suspicious contours on the original image.
    Returns a new BGR image with the overlay applied.
    """
    overlay = original_bgr.copy()
    contours = get_suspicious_contours(edges)

    if not contours:
        return overlay

    # Semi-transparent filled regions
    mask = np.zeros_like(original_bgr)
    cv2.drawContours(mask, contours, -1, (0, 60, 255), thickness=cv2.FILLED)
    overlay = cv2.addWeighted(overlay, 1.0, mask, 0.35, 0)

    # Solid border outline
    cv2.drawContours(overlay, contours, -1, (0, 80, 255), thickness=2)

    # Small dot at centroid of each contour
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(overlay, (cx, cy), 4, (0, 200, 255), -1)

    return overlay
