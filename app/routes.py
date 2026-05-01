"""HTTP routes."""
from __future__ import annotations

import asyncio
import base64
import io
import logging

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from .auth import require_api_key
from .config import get_settings
from .core import model as _model_module
from .core.pipeline import render_overlay, run_inference
from .schemas import HealthResponse, SegmentResponse

logger = logging.getLogger(__name__)
router = APIRouter()

__version__ = "1.0.0"


def _png_b64(rgb_or_gray: np.ndarray) -> str:
    """Encode a numpy image to base64 PNG. Accepts grayscale or RGB."""
    if rgb_or_gray.ndim == 2:
        ok, buf = cv2.imencode(".png", rgb_or_gray)
    else:
        bgr = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model_module._model is not None,
        version=__version__,
    )


@router.post(
    "/v1/segment",
    response_model=SegmentResponse,
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
async def segment(file: UploadFile = File(...)) -> SegmentResponse:
    settings = get_settings()

    # ── Validate content type ──────────────────────────────────────────
    if file.content_type not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type}",
        )

    # ── Validate size ──────────────────────────────────────────────────
    raw = await file.read()
    if len(raw) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.max_image_bytes} bytes",
        )
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload",
        )

    # ── Decode → RGB ndarray (Pillow is more forgiving than cv2) ────────
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")

            if max(im.size) > settings.max_image_dim:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image dimension exceeds {settings.max_image_dim}px",
                )

            # Downscale before converting to ndarray to keep peak RAM bounded.
            # The ROI gets resized to 256x256 anyway — anything past ~1600px
            # only inflates intermediate buffers without improving accuracy.
            longest = max(im.size)
            if longest > settings.process_image_dim:
                scale = settings.process_image_dim / longest
                new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
                im = im.resize(new_size, Image.LANCZOS)
            rgb = np.array(im)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image: {e}",
        )

    h, w = rgb.shape[:2]

    # ── Run inference in a thread (don't block the event loop) ─────────
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_inference, rgb, settings.face_landmarker_path),
            timeout=settings.inference_timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning("Inference timeout for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timed out",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception:
        logger.exception("Inference failure for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        )

    overlay = render_overlay(rgb, result)

    return SegmentResponse(
        image_width=w,
        image_height=h,
        inference_ms=result.inference_ms,
        upper_lip_mask_png_b64=_png_b64(result.upper_lip_mask),
        lower_lip_mask_png_b64=_png_b64(result.lower_lip_mask),
        overlay_png_b64=_png_b64(overlay),
        upper_lip_contour=result.upper_lip_contour,
        lower_lip_contour=result.lower_lip_contour,
        warnings=result.warnings,
    )
