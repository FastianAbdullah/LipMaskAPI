#!/usr/bin/env python3
"""
Lip segmentation client.

Usage:
    python client.py --image path/to/photo.jpg
    python client.py --image photo.jpg --output ./results
    python client.py --image-dir ./test_photos --output ./results

Environment:
    LIP_SEG_API_KEY    your API key

Outputs (per image, written into <output>/<image_stem>/):
    input.png                Original photo (copied)
    upper_lip_mask.png       Binary mask, original resolution
    lower_lip_mask.png       Binary mask, original resolution
    overlay.png              Filled mask + contours visualisation
    contour_coords.json      Polygon coordinates for upper + lower lip
    response.json            Full server response metadata
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
import io
from dotenv import load_dotenv
from PIL import Image as PILImage, ImageOps

load_dotenv()

DEFAULT_URL = "http://72.61.75.134"
DEFAULT_KEY = os.environ.get("LIP_SEG_API_KEY")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print(f"Using API URL: {DEFAULT_URL}, API KEY: {DEFAULT_KEY}")

def _save_b64_png(b64: str, dest: Path) -> None:
    dest.write_bytes(base64.b64decode(b64))


# Resize Image before sending to Network for Optimization.
def _prepare_image(image_path: Path, max_dim: int = 1500) -> tuple[bytes, str]:
    with PILImage.open(image_path) as im:
        im = ImageOps.exif_transpose(im)
        # Convert RGBA/P/LA etc. to RGB before JPEG encoding
        if im.mode != "RGB":
            im = im.convert("RGB")
        
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"

def segment_image(image_path: Path, api_url: str, api_key: str,
                  out_root: Path, timeout: int = 60) -> bool:
    out_dir = out_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  → {image_path.name} ", end="", flush=True)
    t0 = time.perf_counter()

    try:
        img_bytes, mime = _prepare_image(image_path)  # ← resize here
        
        files = {"file": (image_path.name, img_bytes, mime)}  # ← send bytes, not file handle
        headers = {"X-API-Key": api_key}
        r = requests.post(f"{api_url.rstrip('/')}/v1/segment",
                        files=files, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        print(f"[network error] {e}")
        return False

    elapsed = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        try:
            err = r.json().get("detail", r.text)
        except ValueError:
            err = r.text
        print(f"[{r.status_code}] {err}")
        return False

    body = r.json()

    # Save all artefacts
    out_dir_input = out_dir / "input.png"
    out_dir_input.write_bytes(image_path.read_bytes())   # copy original

    _save_b64_png(body["upper_lip_mask_png_b64"], out_dir / "upper_lip_mask.png")
    _save_b64_png(body["lower_lip_mask_png_b64"], out_dir / "lower_lip_mask.png")
    _save_b64_png(body["overlay_png_b64"], out_dir / "overlay.png")

    coords = {
        "image": image_path.name,
        "image_width": body["image_width"],
        "image_height": body["image_height"],
        "upper_lip": body["upper_lip_contour"],
        "lower_lip": body["lower_lip_contour"],
    }
    (out_dir / "contour_coords.json").write_text(json.dumps(coords, indent=2))

    # Full response minus the heavy base64 blobs
    meta = {k: v for k, v in body.items() if not k.endswith("_b64")}
    (out_dir / "response.json").write_text(json.dumps(meta, indent=2))

    warnings = body.get("warnings") or []
    warn_str = f"  warnings={warnings}" if warnings else ""
    print(f"OK  server={body['inference_ms']}ms  total={elapsed:.0f}ms{warn_str}")
    return True


def _guess_mime(p: Path) -> str:
    ext = p.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")


def main() -> int:
    ap = argparse.ArgumentParser(description="Lip segmentation API client")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image file")
    src.add_argument("--image-dir", type=Path, help="Directory of images")
    ap.add_argument("--output", type=Path, default=Path("./lip_seg_results"),
                    help="Output directory (default: ./lip_seg_results)")
    ap.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    ap.add_argument("--timeout", type=int, default=60, help="Per-request timeout (s)")
    args = ap.parse_args()

    if not DEFAULT_KEY:
        print("ERROR: API key required. Set LIP_SEG_API_KEY in .env or as env var",
              file=sys.stderr)
        return 2

    if args.image:
        images = [args.image]
    else:
        images = sorted(
            p for p in args.image_dir.iterdir()
            if p.suffix.lower() in ALLOWED_EXTS
        )
        if not images:
            print(f"No images found in {args.image_dir}", file=sys.stderr)
            return 1

    print(f"API: {args.url}")
    print(f"Output: {args.output.resolve()}")
    print(f"Processing {len(images)} image(s):")

    args.output.mkdir(parents=True, exist_ok=True)

    ok = sum(segment_image(p, args.url, DEFAULT_KEY, args.output, args.timeout)
             for p in images)
    print(f"\nDone. {ok}/{len(images)} succeeded.")
    return 0 if ok == len(images) else 1


if __name__ == "__main__":
    sys.exit(main())
