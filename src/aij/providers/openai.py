"""OpenAI and OpenAI-compatible chat completions."""

from __future__ import annotations

from typing import Any

from aij.errors import ProviderError
from aij.models import Message, ModelResponse, Usage
from aij.providers.base import (
    Provider,
    is_full_endpoint,
    messages_to_dicts,
    normalize_base_url,
)
from aij.providers.pricing import estimate_cost

# OpenAI's reasoning models reject max_tokens and pin temperature to 1.0.
# Detecting them by name keeps the request valid without a probe round-trip.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "o5")


def _is_reasoning_model(model: str) -> bool:
    base = model.split("/")[-1].lower()
    return any(base.startswith(prefix) for prefix in _REASONING_MODEL_PREFIXES)


class OpenAIProvider(Provider):
    name = "openai"

    def _url(self) -> str:
        base = normalize_base_url(
            self.config.base_url or "https://api.openai.com/v1", default_suffix="v1"
        )
        return base if is_full_endpoint(base) else f"{base}/chat/completions"

    def _build_request(self, messages: list[Message]) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {
            "authorization": f"Bearer {self.config.api_key}",
            "content-type": "application/json",
            **self.config.extra_headers,
        }
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages_to_dicts(messages),
        }
        if _is_reasoning_model(self.config.model):
            # Reasoning models: max_completion_tokens, and only the default
            # temperature is accepted.
            body["max_completion_tokens"] = self.config.max_tokens
        else:
            body["temperature"] = self.config.temperature
            body["max_tokens"] = self.config.max_tokens
            if self.config.top_p is not None:
                body["top_p"] = self.config.top_p
        body.update(self.config.extra_body)
        return self._url(), headers, body

    def _parse_response(self, payload: dict[str, Any], latency_ms: float) -> ModelResponse:
        choices = payload.get("choices") or []
        if not choices:
            error = payload.get("error") or {}
            detail = error.get("message") if isinstance(error, dict) else None
            raise _bad_response(self.name, detail or "response contained no choices")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        content = message.get("content")
        if isinstance(content, list):
            # Some compatible servers return content parts instead of a string.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

        # OpenAI signals a safety block three ways: a separate `refusal` field,
        # a `content_filter` finish reason, or an empty message. Every one of
        # them is the endpoint holding the line, which is a refusal verdict --
        # not a transport failure and not a model that failed to answer.
        refusal = message.get("refusal")
        blocked = False
        reason: str | None = None
        if isinstance(refusal, str) and refusal.strip():
            blocked = True
            reason = "api_refusal"
            content = content if content else refusal
        if finish_reason in {"content_filter", "safety"}:
            blocked = True
            reason = reason or f"finish_reason={finish_reason}"
        if content is None:
            content = ""

        return ModelResponse(
            text=str(content),
            model=str(payload.get("model") or self.config.model),
            provider=self.name,
            finish_reason=finish_reason,
            usage=self._estimate_usage(payload),
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=reason,
        )

    def _estimate_usage(self, payload: dict[str, Any]) -> Usage:
        raw = payload.get("usage") or {}
        prompt = int(raw.get("prompt_tokens") or 0)
        completion = int(raw.get("completion_tokens") or 0)
        total = int(raw.get("total_tokens") or prompt + completion)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=estimate_cost(self.name, self.config.model, prompt, completion),
        )


class OpenAICompatibleProvider(OpenAIProvider):
    """Any /v1/chat/completions server: vLLM, LM Studio, Together, Groq, OpenRouter."""

    name = "openai-compatible"

    def _build_request(self, messages: list[Message]) -> tuple[str, dict[str, str], dict[str, Any]]:
        url, headers, body = super()._build_request(messages)
        if not self.config.api_key:
            headers.pop("authorization", None)
        return url, headers, body


def _bad_response(provider: str, detail: str) -> ProviderError:
    return ProviderError(f"{provider}: {detail}", retryable=False)
