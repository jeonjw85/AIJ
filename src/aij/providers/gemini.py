"""Google Gemini generateContent API."""

from __future__ import annotations

from typing import Any

from aij.errors import ProviderError
from aij.models import Message, ModelResponse, Role, Usage
from aij.providers.base import Provider, is_full_endpoint, normalize_base_url
from aij.providers.pricing import estimate_cost


class GeminiProvider(Provider):
    name = "gemini"

    def _url(self) -> str:
        base = normalize_base_url(
            self.config.base_url or "https://generativelanguage.googleapis.com/v1beta",
            default_suffix="v1beta",
        )
        if is_full_endpoint(base):
            return base
        return f"{base}/models/{self.config.model}:generateContent"

    def _build_request(self, messages: list[Message]) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {
            "x-goog-api-key": self.config.api_key or "",
            "content-type": "application/json",
            **self.config.extra_headers,
        }
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        contents = [
            {
                "role": "model" if m.role is Role.ASSISTANT else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role is not Role.SYSTEM
        ]
        generation: dict[str, Any] = {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            generation["topP"] = self.config.top_p
        if self.config.extra_body:
            generation.update(
                self.config.extra_body.get("generationConfig", self.config.extra_body)
            )

        body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return self._url(), headers, body

    def _parse_response(self, payload: dict[str, Any], latency_ms: float) -> ModelResponse:
        candidates = payload.get("candidates") or []
        prompt_block = (payload.get("promptFeedback") or {}).get("blockReason")

        # Gemini withholds the completion via promptFeedback.blockReason (input
        # blocked) or finishReason=SAFETY/PROHIBITED_CONTENT (output blocked).
        # Both are refusals. An empty candidate list with no block reason is the
        # only genuinely broken case.
        if not candidates:
            if prompt_block:
                return ModelResponse(
                    text="",
                    model=str(payload.get("modelVersion") or self.config.model),
                    provider=self.name,
                    finish_reason="SAFETY",
                    usage=self._estimate_usage(payload),
                    latency_ms=latency_ms,
                    blocked=True,
                    block_reason=f"blockReason={prompt_block}",
                )
            raise ProviderError("gemini: no candidates and no block reason")

        candidate = candidates[0] or {}
        finish = candidate.get("finishReason")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        blocked = finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
        return ModelResponse(
            text=text,
            model=str(payload.get("modelVersion") or self.config.model),
            provider=self.name,
            finish_reason=finish,
            usage=self._estimate_usage(payload),
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=f"finishReason={finish}" if blocked else None,
        )

    def _estimate_usage(self, payload: dict[str, Any]) -> Usage:
        raw = payload.get("usageMetadata") or {}
        prompt = int(raw.get("promptTokenCount") or 0)
        completion = int(raw.get("candidatesTokenCount") or 0)
        total = int(raw.get("totalTokenCount") or prompt + completion)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=estimate_cost(self.name, self.config.model, prompt, completion),
        )
