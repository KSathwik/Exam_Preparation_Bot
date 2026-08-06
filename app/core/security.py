"""API-key authentication for sensitive endpoints.

Secure by default: if ``API_AUTH_ENABLED`` is true (the default) and no
``APP_API_KEY`` is configured, a random key is generated once at process
startup and logged so an operator can copy it. Set ``APP_API_KEY`` in ``.env``
for a stable key that survives restarts.

Two entry points are exposed because browsers cannot set custom headers on a
WebSocket handshake:

- ``require_api_key`` — for regular HTTP routes, reads the ``X-API-Key`` header.
- ``require_api_key_ws`` — for the WebSocket route, reads an ``api_key`` query
  parameter instead.
"""

import secrets

from fastapi import HTTPException, Security, WebSocket, status
from fastapi.security import APIKeyHeader
from loguru import logger

from app.core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _resolve_active_key() -> str | None:
    if not settings.api_auth_enabled:
        return None
    if settings.app_api_key:
        return settings.app_api_key
    generated = secrets.token_urlsafe(32)
    logger.warning(
        "No APP_API_KEY configured — generated a temporary key for this process.\n"
        f"  X-API-Key: {generated}\n"
        "  Set APP_API_KEY in .env for a stable key that survives restarts."
    )
    return generated


# Resolved once at import time so the (possibly generated) key is stable for
# the lifetime of the process and logged exactly once at startup.
ACTIVE_API_KEY = _resolve_active_key()


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency guarding sensitive HTTP endpoints behind an API key."""
    if not settings.api_auth_enabled:
        return
    if not api_key or not ACTIVE_API_KEY or not secrets.compare_digest(api_key, ACTIVE_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Provide it via the X-API-Key header.",
        )


async def require_admin_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Guards operator-only endpoints (log tail, metrics) that must not be
    reachable with the same key ``require_api_key`` accepts — that key is
    deliberately embedded in every page load when ``expose_api_key_to_frontend``
    is set, so it isn't a real boundary for anything operator-only. Unlike
    ``require_api_key``, there is no auto-generated fallback: an unset
    ``ADMIN_API_KEY`` disables the route entirely (404) rather than silently
    falling back to the public key.
    """
    if not settings.admin_api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not api_key or not secrets.compare_digest(api_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin API key. Provide it via the X-API-Key header.",
        )


async def require_api_key_ws(websocket: WebSocket) -> bool:
    """WebSocket equivalent of ``require_api_key``.

    Browsers cannot set custom headers when opening a WebSocket, so the key is
    passed as an ``?api_key=`` query parameter instead. Returns ``False`` (and
    closes the socket) when unauthorized — callers must check the return value.
    """
    if not settings.api_auth_enabled:
        return True
    api_key = websocket.query_params.get("api_key")
    if not api_key or not ACTIVE_API_KEY or not secrets.compare_digest(api_key, ACTIVE_API_KEY):
        await websocket.close(code=4401, reason="Missing or invalid API key")
        return False
    return True
