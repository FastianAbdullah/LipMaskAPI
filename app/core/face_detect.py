"""
MediaPipe face landmarker wrapper.

Detects the lip region in an RGB image and returns a padded bounding box
plus the landmark visualisation (kept for debugging — not exposed via API).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Combined outer + inner lip landmark indices (FaceMesh canonical IDs)
LIP_LANDMARK_IDX = sorted({
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
    146, 91, 181, 84, 17, 314, 405, 321, 375, 324, 318, 402, 317, 14, 87, 178, 88, 95,
})

BBox = Tuple[int, int, int, int]


class LipROIDetector:
    """Lazily-loaded singleton wrapping MediaPipe FaceLandmarker."""

    _instance: Optional["LipROIDetector"] = None

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Face landmarker not found: {model_path}")
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    @classmethod
    def get(cls, model_path: Path) -> "LipROIDetector":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def detect(
        self,
        rgb_img: np.ndarray,
        pad_frac: float = 0.05,
    ) -> Optional[Tuple[np.ndarray, BBox]]:
        """
        Returns (roi_crop, bbox) or complete image if no face was found.
        bbox is (x1, y1, x2, y2) in original-image coordinates.
        """
        h, w = rgb_img.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return rgb_img, (0, 0, w, h)

        lm = result.face_landmarks[0]
        xs = [int(lm[i].x * w) for i in LIP_LANDMARK_IDX]
        ys = [int(lm[i].y * h) for i in LIP_LANDMARK_IDX]

        pad = int(max(w, h) * pad_frac)
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w, max(xs) + pad)
        y2 = min(h, max(ys) + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = rgb_img[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
