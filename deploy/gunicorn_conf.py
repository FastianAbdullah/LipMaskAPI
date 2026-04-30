"""
Gunicorn config for the lip-seg API.

CPU inference is single-threaded per request — limit workers to (cores - 1)
and use uvicorn's worker class for async I/O. Each worker loads the model
into its own RAM (~50-100 MB for MobileDeepLabV3 + dependencies).
"""
import multiprocessing
import os

bind = f"{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", max(1, multiprocessing.cpu_count() - 1)))
worker_class = "uvicorn.workers.UvicornWorker"

# Inference can take >5s on a slow VPS — bump the default 30s
timeout = 60
graceful_timeout = 30
keepalive = 5

# Recycle workers periodically to bound memory growth
max_requests = 1000
max_requests_jitter = 100

accesslog = "-"   # stdout — captured by systemd journald
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# Don't preload — we want each worker to load its own copy of the model
# (preload_app=True with PyTorch can cause CUDA / fork issues).
preload_app = False
