"""Response schemas for the segmentation API."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str


class SegmentResponse(BaseModel):
    """All masks returned as base64-encoded PNGs in the original image space."""
    image_width: int
    image_height: int
    inference_ms: float = Field(..., description="Server-side compute time")
    upper_lip_mask_png_b64: str
    lower_lip_mask_png_b64: str
    overlay_png_b64: str = Field(..., description="Original photo + filled mask + contours")
    upper_lip_contour: List[List[int]] = Field(
        ..., description="Douglas-Peucker simplified polygon, [[x,y], ...]",
    )
    lower_lip_contour: List[List[int]]
    warnings: List[str] = []


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
