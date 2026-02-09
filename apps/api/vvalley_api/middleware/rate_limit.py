"""In-process sliding window rate limiter (no external dependencies)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class InMemoryRateLimiter:
    """Simple sliding-window rate limiter keyed by arbitrary string (e.g. IP)."""

    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets[key]
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True

    def reset(self) -> None:
        """Clear all buckets (for testing)."""
        with self._lock:
            self._buckets.clear()
