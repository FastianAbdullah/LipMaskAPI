# Lip Segmentation Client

Send a photo to the lip segmentation API and get back masks, contours, and an overlay — no GPU, no model files needed.

## Install

```bash
pip install requests
```

## Quick start


```bash
python client.py --image photo.jpg --url http://72.61.75.134 --api-key YOUR_KEY
```

Results are saved to `lip_seg_results/<image-name>/`.

## Options

| Flag | Description |
|---|---|
| `--image` | Path to a single image |
| `--image-dir` | Path to a folder of images |
| `--output` | Where to save results (default: `./lip_seg_results`) |
| `--url` | API base URL |
| `--api-key` | Your API key |

## Output files

Each image gets its own folder:

| File | What it is |
|---|---|
| `input.png` | Copy of your original photo |
| `upper_lip_mask.png` | Upper lip binary mask |
| `lower_lip_mask.png` | Lower lip binary mask |
| `overlay.png` | Lips outlined on the original photo |
| `contour_coords.json` | Lip polygon coordinates |
| `response.json` | Timing and metadata |

## Errors

| Code | Cause | Fix |
|---|---|---|
| 401 | Wrong API key | Check your `--api-key` |
| 422 | No face detected | Use a clearer photo with visible lips |
| 413 | Image too large | Keep under 10 MB and 4096 px |
| 504 | Server timeout | Retry |
