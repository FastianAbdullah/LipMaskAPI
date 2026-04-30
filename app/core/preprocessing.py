"""
Preprocessing pipeline: ROI → CLAHE → resize → LBP + GLBP → 5-channel tensor.

Matches the training pipeline 1:1 — any drift between train and inference
preprocessing degrades IoU. Constants here must stay in sync with the notebook.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

INPUT_SIZE: Tuple[int, int] = (256, 256)
LBP_P, LBP_R = 8, 1

# ImageNet stats for RGB; 0.5/0.5 for engineered texture channels
_NORM_MEAN = np.array([0.485, 0.456, 0.406, 0.5, 0.5], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225, 0.5, 0.5], dtype=np.float32)


def apply_clahe(roi_rgb: np.ndarray, clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """Lighting normalisation in LAB space (preserves chroma)."""
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def compute_lbp(gray: np.ndarray, P: int = LBP_P, R: int = LBP_R) -> np.ndarray:
    lbp = local_binary_pattern(gray.astype(np.float64), P, R, method="uniform")
    return (lbp / (lbp.max() + 1e-8) * 255).astype(np.uint8)


def compute_glbp(gray: np.ndarray, P: int = LBP_P, R: int = LBP_R) -> np.ndarray:
    g = gray.astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    lbp = local_binary_pattern(g, P, R, method="uniform")
    glbp = lbp * (grad_mag / (grad_mag.max() + 1e-8))
    return (glbp / (glbp.max() + 1e-8) * 255).astype(np.uint8)


def build_5channel_input(
    roi_rgb: np.ndarray,
    target_size: Tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """
    Full preprocessing — CLAHE → resize → LBP → GLBP → stack → normalise.
    Returns float32 (5, H, W) ready for torch.from_numpy().
    """
    enhanced = apply_clahe(roi_rgb)
    rgb_resized = cv2.resize(enhanced, target_size)
    gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)

    lbp = compute_lbp(gray)
    glbp = compute_glbp(gray)

    stacked = np.dstack([
        rgb_resized,
        lbp[:, :, np.newaxis],
        glbp[:, :, np.newaxis],
    ])  # (H, W, 5) uint8
    normalised = (stacked.astype(np.float32) / 255.0 - _NORM_MEAN) / _NORM_STD
    return normalised.transpose(2, 0, 1)  # (5, H, W)
