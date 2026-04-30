"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .core.face_detect import LipROIDetector
from .core.model import load_model
from .routes import __version__, router

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)-7s %(name)s [%(req_id)s] %(message)s",
    stream=sys.stdout,
)


class ReqIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "req_id"):
            record.req_id = "-"
        return True


for h in logging.root.handlers:
    h.addFilter(ReqIdFilter())

logger = logging.getLogger("app")


# ── Lifespan: load model + face detector once, then serve ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logger.info("Starting %s (env=%s, device=%s)", s.app_name, s.app_env, s.device)
    load_model(s.model_path, device=s.device, num_threads=s.torch_num_threads)
    LipROIDetector.get(s.face_landmarker_path)  # warm up
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Lip Segmentation API",
    version=__version__,
    description="Server-hosted inference for the Mobile DeepLabV3 lip segmentation model.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Per-request ID for log correlation ────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    logger_adapter = logging.LoggerAdapter(logger, {"req_id": req_id})
    logger_adapter.info("%s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger_adapter.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error"},
        )
    response.headers["x-request-id"] = req_id
    return response


app.include_router(router)
