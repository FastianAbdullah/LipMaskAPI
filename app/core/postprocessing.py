"""
Post-processing: morphological cleanup → Gaussian smoothing → teeth exclusion
→ Douglas-Peucker contour extraction.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def gaussian_smooth_mask(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma)
    return ((smoothed > 127).astype(np.uint8)) * 255


def clean_class_mask(raw: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _KERNEL)
    return gaussian_smooth_mask(closed)


def exclude_teeth(lip_mask: np.ndarray, mouth_mask: np.ndarray) -> np.ndarray:
    """Subtract the mouth-opening region from a lip mask to remove teeth bleed."""
    return cv2.bitwise_and(lip_mask, cv2.bitwise_not(mouth_mask))


def extract_dp_contour(mask: np.ndarray, epsilon_frac: float = 0.001) -> Optional[np.ndarray]:
    """Largest external contour, simplified via Douglas-Peucker."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    return cv2.approxPolyDP(largest, epsilon, True)


def upscale_pred_to_full(
    pred_roi: np.ndarray,
    bbox: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Place ROI-resolution prediction back into the original image canvas."""
    h, w = full_shape
    x1, y1, x2, y2 = bbox
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[y1:y2, x1:x2] = cv2.resize(
        pred_roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST,
    )
    return canvas
