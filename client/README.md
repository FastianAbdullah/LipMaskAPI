# Lip Segmentation Client

Send a photo to the lip segmentation API and get back masks, contours, and an overlay.

## First-time Setup

1. **Install Python** — download and install it from [python.org/downloads](https://www.python.org/downloads/). During installation, check the box that says **"Add Python to PATH"**.
2. **Open a terminal** — on Windows, press `Win + R`, type `cmd`, and hit Enter.
3. **Navigate to this folder** — type `cd` followed by the path to this folder, e.g. `cd C:\Users\You\Downloads\client`.
4. **Create a virtual environment** — run `python -m venv .venv`, then activate it with `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux).
5. **Install dependencies** — run `pip install -r requirements.txt`.

You only need to do steps 1–5 once. Next time, just open the terminal, navigate to the folder, activate the venv (step 4), and run the script.

## Quick start

```bash
# Single image
python client.py --image photo.jpg

# Batch — entire folder
python client.py --image-dir ./photos --output ./results
```

Results are saved to `lip_seg_results/<image-name>/` by default.

## Setup

Edit `.env` in this directory with your credentials:

```
LIP_SEG_API_KEY=your_api_key
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--image` | — | Path to a single image |
| `--image-dir` | — | Path to a folder of images |
| `--output` | `./lip_seg_results` | Where to save results |

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
| 401 | Wrong API key | Check `LIP_SEG_API_KEY` in `.env` |
| 422 | No face / lips detected | Use a clearer photo with visible lips |
| 413 | Image too large | Keep under 10 MB and 4096 px |
| 504 | Server timeout | Retry, or increase `--timeout` |