# Lip Segmentation Client

A small Python script that calls the lip segmentation service and saves the results locally. No model files, no GPU, no setup beyond `pip install requests`.

## Setup

```bash
pip install requests
```

That's the only dependency.

## Configuration

Set your API endpoint and key (you'll receive these separately):

```bash
export LIP_SEG_API_URL="https://api.example.com"
export LIP_SEG_API_KEY="your_key_here"
```

(Windows PowerShell)

```powershell
$Env:LIP_SEG_API_URL = "https://api.example.com"
$Env:LIP_SEG_API_KEY = "your_key_here"
```

## Usage

**Single image:**

```bash
python client.py --image photo.jpg
```

**Whole folder:**

```bash
python client.py --image-dir ./test_photos --output ./results
```

**Override the URL / key on the command line:**

```bash
python client.py --image photo.jpg --url https://api.example.com --api-key xxx
```

## Output

For every input image, you get a folder with:

| File                  | What it is                                              |
| --------------------- | ------------------------------------------------------- |
| `input.png`           | Copy of the original photo                              |
| `upper_lip_mask.png`  | Binary mask of the upper lip, original image resolution |
| `lower_lip_mask.png`  | Binary mask of the lower lip, original image resolution |
| `overlay.png`         | Filled mask + contours drawn on the original photo      |
| `contour_coords.json` | Polygon coordinates for both lips                       |
| `response.json`       | Server timing + warnings metadata                       |

## Troubleshooting

- **401 Unauthorized** — check your `LIP_SEG_API_KEY`.
- **422 Unprocessable Entity** — no face detected. Use a clearer photo where the lips are visible.
- **413 Request Entity Too Large** — image bigger than the server limit (10 MB / 4096 px). Resize before sending.
- **504 Gateway Timeout** — server was busy. Retry.

## Performance

Each request typically takes 200–800 ms on a CPU server, plus your network latency. Run a folder of 50 images sequentially in roughly half a minute on a decent connection.
