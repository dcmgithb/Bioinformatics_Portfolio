from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """Validate the X-API-Key header against the API_KEY environment variable.

    Raises ``HTTP 401 Unauthorized`` if the key is absent or does not match.

    Args:
        api_key: The value extracted from the ``X-API-Key`` request header.

    Returns:
        The validated API key string.
    """
    expected: str = os.environ.get("API_KEY", "")
    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
