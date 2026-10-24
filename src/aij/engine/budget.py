"""Async token-bucket rate limiter and a shared cost/request budget."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token bucket. ``rate`` is permits per second; ``burst`` caps the bucket."""

    def __init__(self, rate: float, burst: int = 1) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = max(1.0, float(burst))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            await asyncio.sleep(min(wait, 1.0))


class Budget:
    """Tracks requests, cost, and elapsed time against optional ceilings."""

    def __init__(
        self,
        *,
        max_requests: int | None = None,
        max_cost_usd: float | None = None,
        max_seconds: float | None = None,
    ) -> None:
        self.max_requests = max_requests
        self.max_cost_usd = max_cost_usd
        self.max_seconds = max_seconds
        self.requests = 0
        self.cost_usd = 0.0
        self._started = time.monotonic()
        self._lock = asyncio.Lock()
        self.exhausted_by: str | None = None

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started

    def exhausted(self) -> str | None:
        """Return the ceiling name that is already breached, if any."""
        if self.exhausted_by:
            return self.exhausted_by
        if self.max_requests is not None and self.requests >= self.max_requests:
            self.exhausted_by = "requests"
        elif self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            self.exhausted_by = "cost"
        elif self.max_seconds is not None and self.elapsed_s >= self.max_seconds:
            self.exhausted_by = "time"
        return self.exhausted_by

    async def reserve(self) -> None:
        """Claim one request slot, raising if a ceiling is already hit."""
        async with self._lock:
            reason = self.exhausted()
            if reason is not None:
                from aij.errors import BudgetExceeded

                raise BudgetExceeded(f"budget exhausted by {reason}")
            self.requests += 1

    async def charge(self, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        async with self._lock:
            self.cost_usd += cost_usd
            if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
                self.exhausted_by = "cost"

    async def charge_extra_request(self) -> None:
        """Account for a retry that already went out on the wire.

        ``reserve`` runs before a call; retries happen inside the provider, so
        the scanner reconciles the difference afterwards. This must never raise:
        the request already happened, and refusing to count it would understate
        spend.
        """
        async with self._lock:
            self.requests += 1
            if self.max_requests is not None and self.requests >= self.max_requests:
                self.exhausted_by = self.exhausted_by or "requests"

    def snapshot(self) -> dict[str, float | int | str | None]:
        return {
            "requests": self.requests,
            "cost_usd": round(self.cost_usd, 6),
            "elapsed_s": round(self.elapsed_s, 2),
            "exhausted_by": self.exhausted_by,
            "max_requests": self.max_requests,
            "max_cost_usd": self.max_cost_usd,
            "max_seconds": self.max_seconds,
        }
