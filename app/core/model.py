"""
Mobile DeepLabV3 + MobileNetV3-Large with 5-channel input head.

Architecture must match the training notebook exactly so the .pth state_dict
loads without missing/unexpected keys. Do NOT alter layer names or shapes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

NUM_CLASSES = 4   # 0=bg, 1=upper_lip, 2=lower_lip, 3=mouth_opening
IN_CHANNELS = 5   # RGB + LBP + GLBP

logger = logging.getLogger(__name__)


class MobileDeepLabV3Lip(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        weights = (
            tv_models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            if pretrained_backbone else None
        )
        base = tv_models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=weights, num_classes=21,
        )

        # Widen first conv to 5 channels — RGB weights copied, texture channels zero-init
        old_conv = base.backbone["0"][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained_backbone:
                new_conv.weight[:, :3, :, :] = old_conv.weight
            nn.init.zeros_(new_conv.weight[:, 3:, :, :])
        base.backbone["0"][0] = new_conv

        # Replace classifier head for our class count
        in_feats = base.classifier[4].in_channels
        base.classifier[4] = nn.Conv2d(in_feats, num_classes, kernel_size=1)
        if base.aux_classifier is not None:
            aux_in = base.aux_classifier[4].in_channels
            base.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)["out"]
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        return out


# ── Process-wide singleton ────────────────────────────────────────────────────
_model_lock = Lock()
_model: Optional[MobileDeepLabV3Lip] = None


def load_model(weights_path: Path, device: str = "cpu",
               num_threads: int = 2) -> MobileDeepLabV3Lip:
    """Load checkpoint once. Idempotent and thread-safe."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model

        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        if device == "cpu":
            torch.set_num_threads(num_threads)

        m = MobileDeepLabV3Lip(pretrained_backbone=False)
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("State dict mismatch — missing=%d unexpected=%d",
                           len(missing), len(unexpected))
        m.to(device).eval()
        _model = m
        logger.info("Model loaded from %s on %s", weights_path, device)
        return _model


def get_model() -> MobileDeepLabV3Lip:
    if _model is None:
        raise RuntimeError("Model not loaded — call load_model() at startup")
    return _model
