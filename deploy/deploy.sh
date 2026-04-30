#!/usr/bin/env bash
#
# Deploys lip-seg-api to a fresh Hostinger Ubuntu 22.04+ VPS.
#
# Run as root:
#   bash deploy/deploy.sh
#
# Before running, edit the CONFIG block below.
# This script is idempotent — re-running it updates the install.

set -euo pipefail

# ─── CONFIG ─────────────────────────────────────────────────────────────────
APP_USER="lipseg"
APP_DIR="/var/www/apps/lip-seg-api"
PYTHON_BIN="python3.11"     # adjust if your VPS has a different version
DOMAIN="lip.qubiloo.cloud"  # subdomain for this API
# ────────────────────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
   echo "Run as root (sudo bash deploy/deploy.sh)" >&2
   exit 1
fi

echo "[1/8] Installing system packages..."
apt-get update
apt-get install -y --no-install-recommends \
    "${PYTHON_BIN}" "${PYTHON_BIN}-venv" "${PYTHON_BIN}-dev" \
    build-essential libgl1 libglib2.0-0 \
    nginx curl ca-certificates

echo "[2/8] Creating service user..."
id -u "${APP_USER}" &>/dev/null || useradd --system --home "${APP_DIR}" --shell /usr/sbin/nologin "${APP_USER}"

echo "[3/8] Syncing source to ${APP_DIR}..."
mkdir -p "${APP_DIR}"
rsync -a --delete \
    --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
    --exclude 'tests' --exclude 'client' --exclude '.env' \
    ./ "${APP_DIR}/"

echo "[4/8] Setting up virtualenv..."
if [[ ! -d "${APP_DIR}/.venv" ]]; then
    "${PYTHON_BIN}" -m venv "${APP_DIR}/.venv"
fi
"${APP_DIR}/.venv/bin/pip" install --upgrade pip wheel
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

echo "[5/8] Verifying model files..."
if [[ ! -f "${APP_DIR}/models/mobile_deeplabv3_lip.pth" ]]; then
    echo "  WARNING: models/mobile_deeplabv3_lip.pth is missing." >&2
    echo "  Upload it to the server before starting the service." >&2
fi
if [[ ! -f "${APP_DIR}/models/face_landmarker.task" ]]; then
    echo "  Downloading MediaPipe face_landmarker.task..."
    curl -fsSL -o "${APP_DIR}/models/face_landmarker.task" \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
fi

if [[ ! -f "${APP_DIR}/.env" ]]; then
    echo "  Creating ${APP_DIR}/.env from example. EDIT THIS BEFORE STARTING."
    cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
    NEW_KEY="$(openssl rand -hex 32)"
    sed -i "s|^API_KEYS=.*|API_KEYS=${NEW_KEY}|" "${APP_DIR}/.env"
    echo "  Generated initial API key: ${NEW_KEY}"
    echo "  (saved into ${APP_DIR}/.env — share with client securely)"
fi

chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
chmod 600 "${APP_DIR}/.env"
chmod 600 "${APP_DIR}/models/"*.pth 2>/dev/null || true

echo "[6/8] Installing systemd unit..."
cp "${APP_DIR}/deploy/lip-seg-api.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable lip-seg-api.service
systemctl restart lip-seg-api.service

echo "[7/8] Configuring nginx..."
if [[ -n "${DOMAIN}" ]]; then
    sed "s|<YOUR_DOMAIN>|${DOMAIN}|g" "${APP_DIR}/deploy/nginx.conf" \
        > /etc/nginx/sites-available/lip-seg-api
    ln -sf /etc/nginx/sites-available/lip-seg-api /etc/nginx/sites-enabled/lip-seg-api
    rm -f /etc/nginx/sites-enabled/default
    nginx -t
    systemctl reload nginx
    echo "  Nginx configured for ${DOMAIN}."
    echo "  Run:  sudo certbot --nginx -d ${DOMAIN}   to issue TLS certs."
else
    echo "  DOMAIN not set — skipping nginx step. App listens on 127.0.0.1:8000."
fi

echo "[8/8] Status:"
systemctl --no-pager status lip-seg-api.service | head -n 12 || true
echo
echo "Done. Tail logs with:  journalctl -u lip-seg-api -f"
