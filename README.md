# Lip Segmentation API

Server-hosted inference for the Mobile DeepLabV3 lip segmentation model. Built so that the trained `.pth` weights stay on the server while the client gets a thin CLI that feels exactly like a local script.

## Architecture

```
client.py  ──HTTPS──▶  Nginx  ──▶  Gunicorn (Uvicorn workers)  ──▶  FastAPI  ──▶  Pipeline
                       (TLS,                                                       │
                        rate limit,                                                ▼
                        body size)                                          model.pth (in-memory)
```

- Model loads **once per worker** at startup (lifespan).
- API key auth via `X-API-Key` header, constant-time compare.
- Inference runs in a thread pool (`asyncio.to_thread`) so the event loop stays free.
- Per-request UUID for log correlation.
- Multipart upload, base64 PNG response — single round trip, easy to script.

## Project layout

```
lip-seg-api/
├── app/                       # Server code
│   ├── main.py                # FastAPI app + lifespan + middleware
│   ├── config.py              # pydantic-settings
│   ├── auth.py                # API key dependency
│   ├── routes.py              # /health, /v1/segment
│   ├── schemas.py             # Response models
│   └── core/                  # Pure inference logic (no HTTP)
│       ├── face_detect.py     # MediaPipe wrapper
│       ├── preprocessing.py   # CLAHE + LBP + GLBP + 5-channel
│       ├── model.py           # MobileDeepLabV3Lip + loader
│       ├── postprocessing.py  # Morphology + smoothing + contours
│       └── pipeline.py        # End-to-end orchestrator
│
├── client/
│   ├── client.py              # The CLI you give to the client
│   └── README.md              # Their docs
│
├── deploy/
│   ├── deploy.sh              # One-shot install for fresh Ubuntu VPS
│   ├── lip-seg-api.service    # systemd unit (hardened)
│   ├── nginx.conf             # Reverse proxy + TLS + rate limit
│   └── gunicorn_conf.py       # Worker config
│
├── models/                    # .pth + face_landmarker.task (NOT committed)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Drop your weights in:
# models/mobile_deeplabv3_lip.pth
# models/face_landmarker.task

cp .env.example .env
# generate a key:
echo "API_KEYS=$(openssl rand -hex 32)" >> .env

# run dev server
uvicorn app.main:app --reload --port 8000
```

Then in another terminal:

```bash
export LIP_SEG_API_URL=http://127.0.0.1:8000
export LIP_SEG_API_KEY=<paste from .env>
python client/client.py --image some_face.jpg
```

## Production deployment (Hostinger VPS)

1. SSH in as root.
2. Clone or rsync the repo to the server.
3. Edit the `CONFIG` block in `deploy/deploy.sh` (set your domain).
4. Upload `mobile_deeplabv3_lip.pth` to `/opt/lip-seg-api/models/`.
5. Run:
   ```bash
   sudo bash deploy/deploy.sh
   ```
6. If you set a domain:
   ```bash
   sudo certbot --nginx -d api.your-domain.com
   ```

That's it. Logs:

```bash
journalctl -u lip-seg-api -f
```

## API

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

### `POST /v1/segment`

**Headers:** `X-API-Key: <your key>`
**Body:** multipart `file=<image>`

**Response:**

```json
{
  "image_width": 1024,
  "image_height": 1024,
  "inference_ms": 412.3,
  "upper_lip_mask_png_b64": "iVBOR...",
  "lower_lip_mask_png_b64": "iVBOR...",
  "overlay_png_b64": "iVBOR...",
  "upper_lip_contour": [[x, y], ...],
  "lower_lip_contour": [[x, y], ...],
  "warnings": []
}
```

| Status | Meaning                                  |
| ------ | ---------------------------------------- |
| 200    | OK                                       |
| 400    | Empty / undecodable upload               |
| 401    | Missing or invalid API key               |
| 413    | Image too large (bytes or dimensions)    |
| 415    | Unsupported MIME type                    |
| 422    | No face detected                         |
| 504    | Inference timed out                      |

## Security notes

- `.env` perms are forced to `600`; weights to `600`.
- systemd unit runs as a non-login `lipseg` user with kernel/system hardening.
- Nginx rate-limits to 30 req/min per IP with a 10-request burst.
- API keys are compared in constant time. Rotate by editing `API_KEYS` and `systemctl restart lip-seg-api`.
- `models/*.pth` is in `.gitignore` — don't commit weights.

## Operations

| Task                  | Command                                                              |
| --------------------- | -------------------------------------------------------------------- |
| Restart               | `sudo systemctl restart lip-seg-api`                                 |
| Tail logs             | `journalctl -u lip-seg-api -f`                                       |
| Rotate API key        | edit `/opt/lip-seg-api/.env` → restart                               |
| Update code           | `git pull && sudo bash deploy/deploy.sh`                             |
| Check worker health   | `curl https://<domain>/health`                                       |
| Smoke-test inference  | `python client/client.py --image test.jpg`                           |
