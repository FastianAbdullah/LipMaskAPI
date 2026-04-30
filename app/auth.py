"""API key authentication via X-API-Key header. Constant-time comparison."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency. Raises 401 if the supplied key is invalid."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    valid = get_settings().valid_keys()
    if not valid:
        # Misconfigured server — fail closed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this server",
        )

    # Constant-time comparison against each known key
    for k in valid:
        if hmac.compare_digest(x_api_key, k):
            return x_api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )
