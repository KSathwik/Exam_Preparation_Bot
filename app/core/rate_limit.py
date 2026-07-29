"""Request-rate limiting for expensive, LLM-calling endpoints.

Keyed by client IP. The limit is configurable via ``RATE_LIMIT_PER_MINUTE`` so
it can be tuned (or disabled by setting it very high) without code changes.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

limiter = Limiter(key_func=get_remote_address, default_limits=[])


def limit_expensive():
    """Route decorator applying the configured per-minute request limit."""
    return limiter.limit(f"{settings.rate_limit_per_minute}/minute")
