from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    app_name: str = "lip-seg-api"
    app_env: str = "production"
    log_level: str = "INFO"

    # ── Auth ──────────────────────────────────────────────────────────────
    api_keys: str = Field(default="", description="Comma-separated valid API keys")

    # ── Model ─────────────────────────────────────────────────────────────
    model_path: Path = Path("models/mobile_deeplabv3_lip.pth")
    face_landmarker_path: Path = Path("models/face_landmarker.task")
    device: str = "cpu"  # "cpu" | "cuda"
    torch_num_threads: int = 2

    # ── Inference limits ──────────────────────────────────────────────────
    max_image_bytes: int = 50 * 1024 * 1024        # 10 MB upload cap
    max_image_dim: int = 4096                       # px — reject anything bigger
    process_image_dim: int = 1600                   # px — downscale to this before processing
    allowed_mime_types: List[str] = [
        "image/jpeg", "image/png", "image/webp", "image/bmp",
    ]
    inference_timeout_sec: int = 30

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000

    @field_validator("api_keys")
    @classmethod
    def _strip_keys(cls, v: str) -> str:
        return ",".join(k.strip() for k in v.split(",") if k.strip())

    def valid_keys(self) -> set[str]:
        return set(k for k in self.api_keys.split(",") if k)


@lru_cache
def get_settings() -> Settings:
    return Settings()
