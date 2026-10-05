"""
Lightweight in-process rate limiting for GLM-API.

A sliding-window log limiter, keyed by client IP.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimiter:
    """Sliding-window request counter shared across requests (thread-safe)."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, now: float) -> Tuple[bool, int, float]:
        """Record a request for `key`. Returns (allowed, remaining, retry_after)."""
        cutoff = now - self.window
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= self.limit:
                retry_after = q[0] + self.window - now
                return False, 0, max(0.0, retry_after)

            q.append(now)
            return True, self.limit - len(q), 0.0


def install_rate_limit(app, limiter: RateLimiter, *, protect_prefix: str = "/v1") -> None:
    """Attach `limiter` to FastAPI app as HTTP middleware."""

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        p = request.url.path
        if not (p.startswith(protect_prefix) or p in ("/chat/completions", "/models", "/embeddings")):
            return await call_next(request)

        key = _client_key(request)
        now = time.time()
        allowed, remaining, retry_after = limiter.hit(key, now)

        if not allowed:
            resp = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": (
                            f"Rate limit exceeded: {limiter.limit} requests per "
                            f"{int(limiter.window)}s. Retry in "
                            f"{retry_after:.1f}s."
                        ),
                        "type": "rate_limit_error",
                    }
                },
            )
            resp.headers["Retry-After"] = str(int(retry_after) + 1)
        else:
            resp = await call_next(request)

        resp.headers["X-RateLimit-Limit"] = str(limiter.limit)
        resp.headers["X-RateLimit-Remaining"] = str(remaining)
        resp.headers["X-RateLimit-Reset"] = str(int(now + limiter.window))
        return resp


def _client_key(request: Request) -> str:
    """Best-effort client identity: first X-Forwarded-For hop, else peer IP."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
