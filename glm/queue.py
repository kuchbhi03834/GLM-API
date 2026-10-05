"""
Concurrency & Task Queue for GLM-API.

Maintains a non-blocking queue and bounded concurrency pool so that rapid IDE
requests (e.g. from OpenCode) queue safely and timeout gracefully rather than
deadlocking or throwing 500 errors.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional


class GLMTimeoutError(RuntimeError):
    """Raised when waiting in the GLM request queue times out."""
    pass


class GLMTaskQueue:
    def __init__(self, max_workers: int = 4, timeout: float = 30.0):
        self.max_workers = max(1, max_workers)
        self.timeout = timeout
        self._semaphore = threading.Semaphore(self.max_workers)
        self._active_count = 0
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a slot in the concurrency pool within timeout."""
        effective_timeout = timeout if timeout is not None else self.timeout
        acquired = self._semaphore.acquire(blocking=True, timeout=effective_timeout)
        if not acquired:
            raise GLMTimeoutError(
                f"GLM queue timed out after {effective_timeout:.1f}s. "
                "The server is experiencing high concurrency; please retry."
            )
        with self._lock:
            self._active_count += 1
        return True

    def release(self) -> None:
        """Release a slot back to the concurrency pool."""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
        self._semaphore.release()

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    def stats(self) -> dict:
        """Diagnostic metrics for health checks."""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "active_requests": self._active_count,
                "available_slots": max(0, self.max_workers - self._active_count),
            }


_GLOBAL_QUEUE: Optional[GLMTaskQueue] = None
_QUEUE_LOCK = threading.Lock()


def get_glm_queue(max_workers: int = 4, timeout: float = 30.0) -> GLMTaskQueue:
    global _GLOBAL_QUEUE
    if _GLOBAL_QUEUE is None:
        with _QUEUE_LOCK:
            if _GLOBAL_QUEUE is None:
                workers = int(os.getenv("GLM_WORKERS", str(max_workers)))
                t_out = float(os.getenv("GLM_REQUEST_TIMEOUT", str(timeout)))
                _GLOBAL_QUEUE = GLMTaskQueue(max_workers=workers, timeout=t_out)
    return _GLOBAL_QUEUE
