import multiprocessing
import os

bind = f"{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8003')}"
workers = int(os.environ.get("WEB_CONCURRENCY", max(1, multiprocessing.cpu_count() - 1)))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 60
graceful_timeout = 30
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
preload_app = False
