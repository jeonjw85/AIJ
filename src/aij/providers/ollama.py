"""Local Ollama chat endpoint."""

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


class OllamaProvider(Provider):
    name = "ollama"

    def _url(self) -> str:
        base = normalize_base_url(self.config.base_url or "http://localhost:11434")
        return base if is_full_endpoint(base) else f"{base}/api/chat"

    def _build_request(self, messages: list[Message]) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"content-type": "application/json", **self.config.extra_headers}
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages_to_dicts(messages),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if self.config.top_p is not None:
            body["options"]["top_p"] = self.config.top_p
        body.update(self.config.extra_body)
        return self._url(), headers, body

    def _parse_response(self, payload: dict[str, Any], latency_ms: float) -> ModelResponse:
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ProviderError(f"ollama: {payload.get('error') or 'missing message field'}")
        return ModelResponse(
            text=str(message.get("content") or ""),
            model=str(payload.get("model") or self.config.model),
            provider=self.name,
            finish_reason=payload.get("done_reason"),
            usage=self._estimate_usage(payload),
            latency_ms=latency_ms,
        )

    def _estimate_usage(self, payload: dict[str, Any]) -> Usage:
        prompt = int(payload.get("prompt_eval_count") or 0)
        completion = int(payload.get("eval_count") or 0)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cost_usd=0.0,
        )
