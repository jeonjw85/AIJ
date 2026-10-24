"""Approximate per-million-token pricing for cost estimation.

These are estimates used for budget ceilings only; providers remain the source
of truth for billing. Unknown models fall back to zero so a scan never fails
just because pricing is missing.
"""

from __future__ import annotations

# (input, output, cached_input) USD per 1M tokens
PRICING: dict[str, tuple[float, float, float]] = {
    "openai:gpt-4o": (2.50, 10.00, 1.25),
    "openai:gpt-4o-mini": (0.15, 0.60, 0.075),
    "openai:gpt-4.1": (2.00, 8.00, 0.50),
    "openai:gpt-4.1-mini": (0.40, 1.60, 0.10),
    "openai:o3": (2.00, 8.00, 0.50),
    "openai:o4-mini": (1.10, 4.40, 0.275),
    "anthropic:claude-sonnet-4": (3.00, 15.00, 0.30),
    "anthropic:claude-opus-4": (15.00, 75.00, 1.50),
    "anthropic:claude-3-5-haiku": (0.80, 4.00, 0.08),
    "gemini:gemini-2.5-pro": (1.25, 10.00, 0.31),
    "gemini:gemini-2.5-flash": (0.30, 2.50, 0.075),
    "gemini:gemini-2.0-flash": (0.10, 0.40, 0.025),
}


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    key = f"{provider}:{model}"
    rates = PRICING.get(key)
    if rates is None:
        base = model.split("/")[-1].lower()
        for candidate, candidate_rates in PRICING.items():
            if candidate.split(":", 1)[1].lower() == base:
                rates = candidate_rates
                break
    if rates is None:
        # Prefix match handles dated snapshots like gpt-4o-2024-11-20.
        lowered = model.lower()
        for candidate, candidate_rates in PRICING.items():
            if candidate.split(":", 1)[1].lower() in lowered:
                rates = candidate_rates
                break
    if rates is None:
        return 0.0
    input_rate, output_rate, _ = rates
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
