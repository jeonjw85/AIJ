"""Provider interface and shared HTTP plumbing."""

from __future__ import annotations

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from aij.errors import FatalProviderError, ProviderError
from aij.models import Message, ModelResponse, ProviderConfig, Usage

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# Statuses that mean "this scan cannot succeed": the key is wrong, the model is
# not available to this account, or the account cannot pay. Mapped to a fatal
# error so the scanner stops after the first probe instead of burning the matrix.
FATAL_STATUS = {401, 403, 404}


class Provider(ABC):
    """Sends a message list to a model and normalizes the reply."""

    name: str = "provider"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model

    @abstractmethod
    def _build_request(self, messages: list[Message]) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Return (url, headers, json_body)."""

    @abstractmethod
    def _parse_response(self, payload: dict[str, Any], latency_ms: float) -> ModelResponse:
        """Convert a provider payload into a ModelResponse."""

    @abstractmethod
    def _estimate_usage(self, payload: dict[str, Any]) -> Usage:
        """Extract token usage and cost from a raw payload."""

    async def complete(
        self,
        messages: list[Message],
        *,
        client: httpx.AsyncClient,
        max_retries: int = 3,
    ) -> ModelResponse:
        url, headers, body = self._build_request(messages)
        attempt = 0
        last_error: ProviderError | None = None

        while attempt <= max_retries:
            attempt += 1
            started = time.perf_counter()
            retry_after: float | None = None
            try:
                response = await client.post(url, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                last_error = ProviderError(f"{self.name}: request timed out", retryable=True)
                last_error.__cause__ = exc
            except httpx.HTTPError as exc:
                last_error = ProviderError(f"{self.name}: transport error: {exc}", retryable=True)
            else:
                latency_ms = (time.perf_counter() - started) * 1000
                if response.status_code < 400:
                    payload = _decode_body(response)
                    if payload is None:
                        raise ProviderError(
                            f"{self.name}: response was not JSON "
                            f"(content-type {response.headers.get('content-type', '?')})",
                            status=response.status_code,
                        )
                    if not isinstance(payload, dict):
                        raise ProviderError(
                            f"{self.name}: unexpected JSON root of type {type(payload).__name__}"
                        )
                    parsed = self._parse_response(payload, latency_ms)
                    parsed.attempts = attempt
                    if parsed.usage.total_tokens == 0:
                        parsed.usage = self._estimate_usage(payload)
                    return parsed

                text = _safe_text(response)
                last_error = ProviderError(
                    f"{self.name}: HTTP {response.status_code}: {text[:400]}",
                    status=response.status_code,
                    retryable=response.status_code in RETRY_STATUS,
                    body=text[:2000],
                )
                if response.status_code not in RETRY_STATUS:
                    raise _classify(last_error)
                retry_after = _retry_after_seconds(response)

            if attempt > max_retries:
                break
            delay = min(30.0, 0.75 * (2 ** (attempt - 1)))
            delay *= 0.75 + random.random() * 0.5
            await asyncio.sleep(retry_after if retry_after is not None else delay)

        assert last_error is not None
        raise last_error


def _classify(error: ProviderError) -> ProviderError:
    """Upgrade a non-retryable status into a fatal one when it is unfixable."""
    if error.status in FATAL_STATUS:
        return FatalProviderError(
            error.message, status=error.status, retryable=False, body=error.body
        )
    return error


def _decode_body(response: httpx.Response) -> dict[str, Any] | None:
    """Parse a completion body, tolerating SSE that ignores stream=false.

    Some gateways and proxies return text/event-stream even for a non-streaming
    request. The useful payload is the last ``data:`` frame, which carries the
    same shape as a normal completion.
    """
    try:
        parsed = response.json()
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    text = _safe_text(response)
    if not text.lstrip().startswith(("data:", "event:")):
        return None

    last: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            frame = json.loads(chunk)
        except ValueError:
            continue
        # Prefer a frame that actually carries choices.
        if isinstance(frame, dict) and (frame.get("choices") or last is None):
            last = frame
    return last


def _safe_text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:
        return "<unreadable body>"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def messages_to_dicts(messages: list[Message]) -> list[dict[str, str]]:
    return [m.as_dict() for m in messages]


_ENDPOINT_TAILS = {"messages", "chat", "completions", "embeddings", "generatecontent"}


def is_full_endpoint(url: str) -> bool:
    """True when the URL already names a concrete API operation."""
    tail = url.strip().rstrip("/").rsplit("/", 1)[-1].lower()
    return tail in _ENDPOINT_TAILS or (":generatecontent" in url.lower())


def normalize_base_url(url: str, *, default_suffix: str = "") -> str:
    """Make a user-supplied base URL usable.

    Operators paste whatever their provider or proxy handed them: sometimes with
    a trailing slash, sometimes with `/v1`, sometimes without. A missing `/v1`
    produces a bare 404 that looks like an outage, so add the version segment
    when the URL has none of its own.
    """
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return cleaned
    # Already versioned (…/v1, …/v1beta, …/v2) or a full operation path: as-is.
    tail = cleaned.rsplit("/", 1)[-1].lower()
    if tail.startswith("v") and tail[1:2].isdigit():
        return cleaned
    if is_full_endpoint(cleaned):
        return cleaned
    if default_suffix:
        return f"{cleaned}/{default_suffix}"
    return cleaned
