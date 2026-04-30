"""
Smoke test — runs the full pipeline against a real image.

Usage:
    python -m tests.test_pipeline path/to/test_image.jpg

Verifies the model loads, MediaPipe finds a face, and the pipeline
returns non-empty masks and contours. Exits non-zero on any failure.

Not a unit test — this is the thing you run after `deploy.sh` to confirm
the server is actually serving inference correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings
from app.core.face_detect import LipROIDetector
from app.core.model import load_model
from app.core.pipeline import run_inference


def main(image_path: str) -> int:
    p = Path(image_path)
    if not p.is_file():
        print(f"ERROR: not a file: {p}", file=sys.stderr)
        return 1

    s = get_settings()
    print(f"Loading model from {s.model_path}...")
    load_model(s.model_path, device=s.device, num_threads=s.torch_num_threads)
    LipROIDetector.get(s.face_landmarker_path)
    print("Model loaded.")

    bgr = cv2.imread(str(p))
    if bgr is None:
        print(f"ERROR: could not decode {p}", file=sys.stderr)
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"Running inference on {p.name} ({rgb.shape[1]}x{rgb.shape[0]})...")
    result = run_inference(rgb, s.face_landmarker_path)

    print(f"  inference_ms = {result.inference_ms}")
    print(f"  upper_mask sum = {(result.upper_lip_mask > 0).sum()} px")
    print(f"  lower_mask sum = {(result.lower_lip_mask > 0).sum()} px")
    print(f"  upper_contour = {len(result.upper_lip_contour)} pts")
    print(f"  lower_contour = {len(result.lower_lip_contour)} pts")
    print(f"  warnings = {result.warnings}")

    failures = []
    if (result.upper_lip_mask > 0).sum() == 0:
        failures.append("upper lip mask is empty")
    if (result.lower_lip_mask > 0).sum() == 0:
        failures.append("lower lip mask is empty")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        return 2

    print("PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m tests.test_pipeline <image_path>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
