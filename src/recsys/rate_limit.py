"""Token bucket rate limiter (cross-pollinated from Spotify/DoorDash backpressure).

Pure Python, deterministic, no external dependency.
"""

from __future__ import annotations


class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float) -> None:
        if capacity <= 0 or refill_rate <= 0:
            raise ValueError("capacity and refill_rate must be > 0")
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last = 0.0

    def allow(self, n: float = 1.0, *, now: float | None = None) -> bool:
        if n <= 0:
            raise ValueError("n must be > 0")
        t = now if now is not None else self._now()
        elapsed = max(0.0, t - self.last)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last = t
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def _now(self) -> float:
        import time

        return time.monotonic()
