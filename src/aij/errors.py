"""Exception hierarchy for AIJ."""

from __future__ import annotations

from typing import Any


class AIJError(Exception):
    """Base class for every error raised by AIJ."""


class ConfigError(AIJError):
    """Invalid or incomplete configuration."""


class CorpusError(AIJError):
    """Invalid objective or technique definition."""


class ProviderError(AIJError):
    """A model endpoint failed to produce a usable response."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable
        self.body = body

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.status is not None:
            return f"{self.message} (status={self.status})"
        return self.message


class FatalProviderError(ProviderError):
    """An error no amount of retrying or probing will fix.

    Bad credentials, a missing model, or an unpaid account fail identically for
    every one of the hundred-odd probes in a scan. Continue and the operator
    gets a wall of errors and an empty database; abort immediately with the real
    reason instead.
    """


class BudgetExceeded(AIJError):
    """The scan hit a configured request, cost, or time ceiling."""


class DetectorError(AIJError):
    """A detector could not reach a verdict."""


class StoreError(AIJError):
    """Persistence layer failure."""
