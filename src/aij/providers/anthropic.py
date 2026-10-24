"""Anthropic Messages API."""

from __future__ import annotations

from typing import Any

from aij.errors import ProviderError
from aij.models import Message, ModelResponse, Role, Usage
from aij.providers.base import Provider, is_full_endpoint, normalize_base_url
from aij.providers.pricing import estimate_cost

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def _url(self) -> str:
        base = normalize_base_url(
            self.config.base_url or "https://api.anthropic.com/v1", default_suffix="v1"
        )
        return base if is_full_endpoint(base) else f"{base}/messages"

    def _build_request(self, messages: list[Message]) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {
            "x-api-key": self.config.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            **self.config.extra_headers,
        }
        # Anthropic takes the system prompt out of band and requires the
        # conversation to start with a user turn.
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        conversation = [m for m in messages if m.role is not Role.SYSTEM]
        while conversation and conversation[0].role is not Role.USER:
            conversation = conversation[1:]

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.as_dict() for m in conversation],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if self.config.top_p is not None:
            body["top_p"] = self.config.top_p
        body.update(self.config.extra_body)
        return self._url(), headers, body

    def _parse_response(self, payload: dict[str, Any], latency_ms: float) -> ModelResponse:
        blocks = payload.get("content") or []
        if not isinstance(blocks, list):
            raise ProviderError("anthropic: malformed content field")
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        stop_reason = payload.get("stop_reason")

        # A refusal is a legitimate answer: stop_reason=refusal, or an empty
        # content list when the request was blocked upstream. Only a payload
        # carrying an explicit error should raise.
        blocked = stop_reason == "refusal" or not blocks
        if not text and not blocked:
            error = payload.get("error") or {}
            detail = error.get("message") if isinstance(error, dict) else None
            raise ProviderError(f"anthropic: {detail or 'empty content'}")
        return ModelResponse(
            text=text,
            model=str(payload.get("model") or self.config.model),
            provider=self.name,
            finish_reason=stop_reason,
            usage=self._estimate_usage(payload),
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=f"stop_reason={stop_reason}" if stop_reason == "refusal" else None,
        )

    def _estimate_usage(self, payload: dict[str, Any]) -> Usage:
        raw = payload.get("usage") or {}
        prompt = int(raw.get("input_tokens") or 0)
        completion = int(raw.get("output_tokens") or 0)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cost_usd=estimate_cost(self.name, self.config.model, prompt, completion),
        )
