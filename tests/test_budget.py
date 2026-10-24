"""Rate limiter and budget enforcement."""

from __future__ import annotations

import asyncio
import time

import pytest

from aij.engine.budget import Budget, RateLimiter
from aij.errors import BudgetExceeded


async def test_limiter_allows_burst() -> None:
    limiter = RateLimiter(rate=10.0, burst=5)
    started = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    assert time.monotonic() - started < 0.2


async def test_limiter_throttles_beyond_burst() -> None:
    limiter = RateLimiter(rate=20.0, burst=1)
    started = time.monotonic()
    for _ in range(4):
        await limiter.acquire()
    # 3 extra permits at 20/s ≈ 0.15s.
    assert time.monotonic() - started >= 0.1


def test_limiter_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(rate=0)


async def test_budget_counts_requests() -> None:
    budget = Budget(max_requests=3)
    for _ in range(3):
        await budget.reserve()
    assert budget.requests == 3
    with pytest.raises(BudgetExceeded, match="requests"):
        await budget.reserve()


async def test_budget_cost_ceiling() -> None:
    budget = Budget(max_cost_usd=0.01)
    await budget.charge(0.006)
    assert budget.exhausted() is None
    await budget.charge(0.006)
    assert budget.exhausted() == "cost"
    with pytest.raises(BudgetExceeded, match="cost"):
        await budget.reserve()


async def test_budget_time_ceiling() -> None:
    budget = Budget(max_seconds=0.01)
    await asyncio.sleep(0.05)
    assert budget.exhausted() == "time"
    with pytest.raises(BudgetExceeded):
        await budget.reserve()


async def test_budget_unlimited_never_exhausts() -> None:
    budget = Budget()
    for _ in range(100):
        await budget.reserve()
    await budget.charge(1000.0)
    assert budget.exhausted() is None


async def test_budget_latches_first_reason() -> None:
    budget = Budget(max_requests=1)
    await budget.reserve()
    assert budget.exhausted() == "requests"

    await budget.charge(1.0)
    budget.max_cost_usd = 0.5
    assert budget.exhausted() == "requests"


async def test_budget_ignores_zero_cost() -> None:
    budget = Budget(max_cost_usd=0.001)
    await budget.charge(0.0)
    assert budget.cost_usd == 0.0
    assert budget.exhausted() is None


async def test_concurrent_reservations_are_exact() -> None:
    budget = Budget(max_requests=50)
    await asyncio.gather(*(budget.reserve() for _ in range(50)))
    assert budget.requests == 50
    with pytest.raises(BudgetExceeded):
        await budget.reserve()


def test_snapshot_shape() -> None:
    snap = Budget(max_requests=5, max_cost_usd=1.0).snapshot()
    assert snap["requests"] == 0
    assert snap["max_requests"] == 5
    assert "elapsed_s" in snap
