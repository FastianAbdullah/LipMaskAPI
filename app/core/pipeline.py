"""
End-to-end inference pipeline.

  RGB image
    → MediaPipe ROI crop
    → 5-channel preprocessing
    → DeepLabV3 forward pass
    → upscale prediction to full image
    → post-processing (morphology + smoothing + teeth exclusion)
    → Douglas-Peucker contour extraction
    → result dict

Pure data in, pure data out — no I/O. The HTTP layer handles encoding.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .face_detect import LipROIDetector
from .model import get_model
from .postprocessing import (
    clean_class_mask,
    exclude_teeth,
    extract_dp_contour,
)
from .preprocessing import INPUT_SIZE, build_5channel_input

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    upper_lip_mask: np.ndarray         # uint8 {0,255}
    lower_lip_mask: np.ndarray         # uint8 {0,255}
    upper_lip_contour: list            # [[x,y], ...]
    lower_lip_contour: list            # [[x,y], ...]
    image_shape: tuple                 # (H, W)
    inference_ms: float
    face_detected: bool = True
    warnings: list = field(default_factory=list)


def _contour_to_list(c: Optional[np.ndarray]) -> list:
    if c is None:
        return []
    return c.reshape(-1, 2).astype(int).tolist()


def run_inference(rgb_image: np.ndarray, face_landmarker_path: Path) -> SegmentationResult:
    """
    Run full pipeline on an RGB image.

    Raises:
        ValueError: if no face / lip region is detectable.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected RGB image of shape (H, W, 3)")

    t0 = time.perf_counter()
    h, w = rgb_image.shape[:2]
    warnings: list[str] = []
    face_detected = True

    # 1. ROI detection — with explicit fallback for pre-cropped lip images.
    detector = LipROIDetector.get(face_landmarker_path)
    detection = detector.detect(rgb_image)
    if detection is None:
        # MediaPipe couldn't find a face — most commonly because the upload
        # is already a tight lip crop. Running the model on the un-padded
        # full image makes it predict "lip" pixels right at the image
        # boundary (visible as "wings" at mouth corners after upsample).
        # Pad with replicated border so the model has skin-like context
        # surrounding the lip.
        face_detected = False
        # warnings.append("face_not_detected_used_lip_crop_fallback")
        # logger.warning(
        #     "No face detected; falling back to padded full-image inference"
        #     " (image=%dx%d)", w, h,
        # )
        pad = max(20, int(0.15 * min(h, w)))
        roi = cv2.copyMakeBorder(
            rgb_image, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE,
        )
        bbox = None
    else:
        roi, bbox = detection
        pad = 0

    # 2. Preprocessing → 5-ch tensor
    tensor = build_5channel_input(roi, target_size=INPUT_SIZE)
    inp = torch.from_numpy(tensor).unsqueeze(0)

    # 3. Inference
    model = get_model()
    device = next(model.parameters()).device
    inp = inp.to(device)
    roi_h, roi_w = roi.shape[:2]
    with torch.no_grad():
        logits = model(inp)
        # Bilinear-upsample logits to ROI resolution *before* argmax.
        # Argmax-then-NEAREST resize produces pixel blocks at the upsample
        # ratio; bilinear-then-argmax gives sub-pixel-smooth boundaries.
        logits_roi = F.interpolate(
            logits, size=(roi_h, roi_w), mode="bilinear", align_corners=False,
        )
        pred_roi = torch.argmax(logits_roi, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 4. Place ROI-sized prediction into the full-image canvas
    pred_full = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        # Lip-crop fallback: ROI is the padded full image; crop padding off.
        pred_full[:, :] = pred_roi[pad:pad + h, pad:pad + w]
    else:
        x1, y1, x2, y2 = bbox
        pred_full[y1:y2, x1:x2] = pred_roi

    # 5. Post-process per class
    upper_raw = ((pred_full == 1) * 255).astype(np.uint8)
    lower_raw = ((pred_full == 2) * 255).astype(np.uint8)
    mouth_raw = ((pred_full == 3) * 255).astype(np.uint8)

    upper_clean = clean_class_mask(upper_raw)
    lower_clean = clean_class_mask(lower_raw)
    mouth_clean = clean_class_mask(mouth_raw)

    upper = exclude_teeth(upper_clean, mouth_clean)
    lower = exclude_teeth(lower_clean, mouth_clean)

    # 6. Contours
    upper_dp = extract_dp_contour(upper)
    lower_dp = extract_dp_contour(lower)

    if upper_dp is None:
        warnings.append("upper_lip_contour_empty")
    if lower_dp is None:
        warnings.append("lower_lip_contour_empty")

    elapsed = (time.perf_counter() - t0) * 1000.0

    return SegmentationResult(
        upper_lip_mask=upper,
        lower_lip_mask=lower,
        upper_lip_contour=_contour_to_list(upper_dp),
        lower_lip_contour=_contour_to_list(lower_dp),
        image_shape=(h, w),
        inference_ms=round(elapsed, 1),
        face_detected=face_detected,
        warnings=warnings,
    )


def render_overlay(rgb_image: np.ndarray, result: SegmentationResult) -> np.ndarray:
    """Filled mask + contour visualisation for client convenience."""
    overlay = rgb_image.copy().astype(np.float32)
    green = np.zeros_like(overlay); green[result.upper_lip_mask > 0] = [0, 220, 80]
    blue = np.zeros_like(overlay); blue[result.lower_lip_mask > 0] = [80, 80, 255]
    blended = np.clip(overlay * 0.55 + green * 0.45 + blue * 0.45, 0, 255).astype(np.uint8)

    if result.upper_lip_contour:
        pts = np.array(result.upper_lip_contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(blended, [pts], True, (0, 220, 80), 2)
    if result.lower_lip_contour:
        pts = np.array(result.lower_lip_contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(blended, [pts], True, (80, 80, 255), 2)

    return blended
