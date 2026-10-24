"""Provider registry."""

from __future__ import annotations

from aij.errors import ConfigError
from aij.models import ProviderConfig
from aij.providers.anthropic import AnthropicProvider
from aij.providers.base import Provider
from aij.providers.gemini import GeminiProvider
from aij.providers.ollama import OllamaProvider
from aij.providers.openai import OpenAICompatibleProvider, OpenAIProvider

PROVIDERS: dict[str, type[Provider]] = {
    "openai": OpenAIProvider,
    "openai-compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def build_provider(config: ProviderConfig) -> Provider:
    try:
        cls = PROVIDERS[config.kind]
    except KeyError as exc:
        raise ConfigError(
            f"unknown provider kind {config.kind!r}; available: {', '.join(sorted(PROVIDERS))}"
        ) from exc
    return cls(config)


__all__ = ["PROVIDERS", "Provider", "build_provider"]
