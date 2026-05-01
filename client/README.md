# Lip Segmentation Client

Send a photo to the lip segmentation API and get back masks, contours, and an overlay.

## Install

```bash
pip install requests python-dotenv
```

## Setup

Edit `.env` in this directory with your credentials:

```
LIP_SEG_API_KEY=your_api_key
```

## Quick start

```bash
# Single image (credentials from .env)
python client.py --image photo.jpg

# Batch — entire folder
python client.py --image-dir ./photos --output ./results
```

Results are saved to `lip_seg_results/<image-nam e>/` by default.

## Options

| Flag | Default | Description |
|---|---|---|
| `--image` | — | Path to a single image |
| `--image-dir` | — | Path to a folder of images |
| `--output` | `./lip_seg_results` | Where to save results |
| `--timeout` | `60` | Per-request timeout in seconds |

## Output files

Each image gets its own subfolder inside `<output>/`:

| File | What it is |
|---|---|
| `input.png` | Copy of your original photo |
| `upper_lip_mask.png` | Upper lip binary mask (original resolution) |
| `lower_lip_mask.png` | Lower lip binary mask (original resolution) |
| `overlay.png` | Lips outlined on the original photo |
| `contour_coords.json` | Lip polygon coordinates (upper + lower) |
| `response.json` | Timing and metadata (no image blobs) |

## Errors

| Code | Cause | Fix |
|---|---|---|
| 401 | Wrong API key | Check `LIP_SEG_API_KEY` in `.env`|
| 422 | No face / lips detected | Use a clearer photo with visible lips |
| 413 | Image too large | Keep under 10 MB and 4096 px |
| 504 | Server timeout | Retry, or increase `--timeout` |
