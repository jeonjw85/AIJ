"""Provider request shaping and response parsing."""

from __future__ import annotations

import pytest
from tests.conftest import MockTarget

from aij.errors import ProviderError
from aij.models import Message, ProviderConfig, Role
from aij.providers import build_provider
from aij.providers.pricing import estimate_cost

MESSAGES = [
    Message(role=Role.SYSTEM, content="You are terse."),
    Message(role=Role.USER, content="hello"),
]


def config(kind: str, **overrides) -> ProviderConfig:
    defaults = dict(
        kind=kind,
        model="test-model",
        api_key="test-key",
        max_tokens=128,
        temperature=0.5,
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def test_openai_request_shape() -> None:
    provider = build_provider(config("openai"))
    url, headers, body = provider._build_request(MESSAGES)
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-key"
    assert body["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hello"},
    ]
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.5


def test_openai_parses_content_parts() -> None:
    provider = build_provider(config("openai"))
    parsed = provider._parse_response(
        {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "hello "},
                            {"type": "text", "text": "world"},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        latency_ms=42.0,
    )
    assert parsed.text == "hello world"
    assert parsed.finish_reason == "stop"


def test_openai_raises_on_missing_choices() -> None:
    provider = build_provider(config("openai"))
    with pytest.raises(ProviderError, match="no choices"):
        provider._parse_response({"choices": []}, latency_ms=1.0)


def test_anthropic_hoists_system_and_drops_leading_assistant() -> None:
    provider = build_provider(config("anthropic"))
    messages = [
        Message(role=Role.SYSTEM, content="Be terse."),
        Message(role=Role.ASSISTANT, content="Prefill"),
        Message(role=Role.USER, content="hi"),
    ]
    url, headers, body = provider._build_request(messages)
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "Be terse."
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_parses_usage() -> None:
    provider = build_provider(config("anthropic"))
    parsed = provider._parse_response(
        {
            "model": "claude-test",
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        latency_ms=1.0,
    )
    assert parsed.text == "hi there"
    assert parsed.usage.total_tokens == 15


def test_gemini_maps_roles_and_system() -> None:
    provider = build_provider(config("gemini"))
    messages = [
        Message(role=Role.SYSTEM, content="Be terse."),
        Message(role=Role.USER, content="hi"),
        Message(role=Role.ASSISTANT, content="hello"),
        Message(role=Role.USER, content="again"),
    ]
    url, headers, body = provider._build_request(messages)
    assert "models/test-model:generateContent" in url
    assert headers["x-goog-api-key"] == "test-key"
    assert body["systemInstruction"]["parts"][0]["text"] == "Be terse."
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]


def test_gemini_prompt_block_is_a_refusal() -> None:
    """A safety block is a refusal verdict, never a scan error."""
    provider = build_provider(config("gemini"))
    parsed = provider._parse_response(
        {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, latency_ms=1.0
    )
    assert parsed.blocked is True
    assert parsed.block_reason == "blockReason=SAFETY"


def test_ollama_request_shape() -> None:
    provider = build_provider(config("ollama"))
    url, _headers, body = provider._build_request(MESSAGES)
    assert url.endswith("/api/chat")
    assert body["stream"] is False
    assert body["options"]["num_predict"] == 128


def test_estimate_cost_matches_known_model() -> None:
    cost = estimate_cost("openai", "gpt-4o", 1_000_000, 0)
    assert cost == pytest.approx(2.50)


def test_estimate_cost_handles_dated_snapshot() -> None:
    cost = estimate_cost("openai", "gpt-4o-2024-11-20", 1_000_000, 0)
    assert cost == pytest.approx(2.50)


def test_estimate_cost_unknown_model_is_zero() -> None:
    assert estimate_cost("openai", "some-unlisted-model", 1000, 1000) == 0.0


async def test_extra_headers_are_sent(mock_target: MockTarget) -> None:
    import httpx

    seen: dict[str, str] = {}

    real_post = httpx.AsyncClient.post

    async def spy(self, url, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs.get("headers") or {})
        return await real_post(self, url, **kwargs)

    cfg = config(
        "openai-compatible",
        base_url=mock_target.url,
        extra_headers={"x-custom": "value"},
    )
    provider = build_provider(cfg)
    mock_target.responder = lambda messages: "ok"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(httpx.AsyncClient, "post", spy)
    try:
        async with httpx.AsyncClient() as client:
            await provider.complete([Message(role=Role.USER, content="hi")], client=client)
    finally:
        monkeypatch.undo()
    assert seen.get("x-custom") == "value"


async def test_openai_compatible_without_key_omits_auth(mock_target: MockTarget) -> None:
    import httpx

    seen: dict[str, str] = {}
    real_post = httpx.AsyncClient.post

    async def spy(self, url, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs.get("headers") or {})
        return await real_post(self, url, **kwargs)

    cfg = ProviderConfig(kind="openai-compatible", model="m", base_url=mock_target.url)
    provider = build_provider(cfg)
    mock_target.responder = lambda messages: "ok"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(httpx.AsyncClient, "post", spy)
    try:
        async with httpx.AsyncClient() as client:
            await provider.complete([Message(role=Role.USER, content="hi")], client=client)
    finally:
        monkeypatch.undo()
    assert "authorization" not in seen


# -- safety blocks are refusals, not errors ----------------------------------


def test_openai_refusal_field_is_a_block() -> None:
    provider = build_provider(config("openai"))
    parsed = provider._parse_response(
        {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"content": None, "refusal": "I can't help with that."},
                    "finish_reason": "stop",
                }
            ],
        },
        latency_ms=1.0,
    )
    assert parsed.blocked is True
    assert "I can't help" in parsed.text


def test_openai_content_filter_is_a_block() -> None:
    provider = build_provider(config("openai"))
    parsed = provider._parse_response(
        {
            "model": "gpt-test",
            "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
        },
        latency_ms=1.0,
    )
    assert parsed.blocked is True
    assert parsed.finish_reason == "content_filter"


def test_openai_normal_answer_is_not_blocked() -> None:
    provider = build_provider(config("openai"))
    parsed = provider._parse_response(
        {
            "model": "gpt-test",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        },
        latency_ms=1.0,
    )
    assert parsed.blocked is False


def test_anthropic_refusal_stop_reason_is_a_block() -> None:
    provider = build_provider(config("anthropic"))
    parsed = provider._parse_response(
        {"model": "claude-test", "content": [], "stop_reason": "refusal"}, latency_ms=1.0
    )
    assert parsed.blocked is True
    assert parsed.block_reason == "stop_reason=refusal"


def test_anthropic_empty_content_without_error_is_a_block() -> None:
    """An empty content list is a refusal, not a transport failure."""
    provider = build_provider(config("anthropic"))
    parsed = provider._parse_response(
        {"model": "claude-test", "content": [], "stop_reason": None}, latency_ms=1.0
    )
    assert parsed.blocked is True


def test_gemini_prompt_block_is_a_refusal_not_an_exception() -> None:
    provider = build_provider(config("gemini"))
    parsed = provider._parse_response(
        {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, latency_ms=1.0
    )
    assert parsed.blocked is True
    assert parsed.block_reason == "blockReason=SAFETY"


def test_gemini_safety_finish_reason_is_a_block() -> None:
    provider = build_provider(config("gemini"))
    parsed = provider._parse_response(
        {
            "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
        },
        latency_ms=1.0,
    )
    assert parsed.blocked is True


def test_gemini_unexplained_empty_candidates_still_raises() -> None:
    """No candidates and no block reason is genuinely broken; do not mask it."""
    provider = build_provider(config("gemini"))
    with pytest.raises(ProviderError, match="no candidates"):
        provider._parse_response({"candidates": []}, latency_ms=1.0)


def test_reasoning_model_uses_max_completion_tokens() -> None:
    """o-series rejects max_tokens and pins temperature; a naive request 400s."""
    provider = build_provider(config("openai", model="o3-mini"))
    _, _, body = provider._build_request(MESSAGES)
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 128
    assert "temperature" not in body


def test_dated_reasoning_snapshot_detected() -> None:
    provider = build_provider(config("openai", model="o3-mini-2025-01-31"))
    _, _, body = provider._build_request(MESSAGES)
    assert body["max_completion_tokens"] == 128


def test_reasoning_model_behind_namespace_detected() -> None:
    provider = build_provider(config("openai", model="openai/o4-mini"))
    _, _, body = provider._build_request(MESSAGES)
    assert body["max_completion_tokens"] == 128


def test_regular_model_keeps_max_tokens_and_temperature() -> None:
    provider = build_provider(config("openai", model="gpt-4o-mini"))
    _, _, body = provider._build_request(MESSAGES)
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.5
    assert "max_completion_tokens" not in body


def test_model_named_o1_lookalike_is_not_mistaken() -> None:
    """A model starting with 'ol' or similar must keep standard parameters."""
    provider = build_provider(config("openai", model="olmo-7b"))
    _, _, body = provider._build_request(MESSAGES)
    assert body["max_tokens"] == 128


# -- base URL normalization --------------------------------------------------


def test_base_url_without_version_gets_v1() -> None:
    """Operators paste proxy roots; a bare 404 is indistinguishable from an outage."""
    provider = build_provider(
        config("anthropic", base_url="https://proxy.example.com/apps/anthropic")
    )
    assert provider._url() == "https://proxy.example.com/apps/anthropic/v1/messages"


def test_base_url_with_v1_is_untouched() -> None:
    provider = build_provider(config("anthropic", base_url="https://api.anthropic.com/v1"))
    assert provider._url() == "https://api.anthropic.com/v1/messages"


def test_base_url_trailing_slash_is_trimmed() -> None:
    provider = build_provider(config("openai", base_url="https://api.openai.com/v1/"))
    assert provider._url() == "https://api.openai.com/v1/chat/completions"


def test_gemini_gets_v1beta() -> None:
    provider = build_provider(config("gemini", base_url="https://proxy.example.com"))
    assert provider._url().startswith("https://proxy.example.com/v1beta/models/")


def test_ollama_adds_no_version_segment() -> None:
    provider = build_provider(config("ollama", base_url="http://localhost:11434"))
    assert provider._url() == "http://localhost:11434/api/chat"


def test_base_url_already_an_endpoint_is_not_doubled() -> None:
    """Pasting the full endpoint path must not produce a doubled suffix."""
    provider = build_provider(config("openai", base_url="https://x.dev/v1/chat/completions"))
    assert provider._url() == "https://x.dev/v1/chat/completions"


def test_anthropic_full_endpoint_not_doubled() -> None:
    provider = build_provider(config("anthropic", base_url="https://x.dev/v1/messages"))
    assert provider._url() == "https://x.dev/v1/messages"


def test_v2_style_version_is_respected() -> None:
    provider = build_provider(config("openai", base_url="https://x.dev/v2"))
    assert provider._url() == "https://x.dev/v2/chat/completions"


# -- transport edge cases ----------------------------------------------------


async def test_sse_response_is_decoded(mock_target: MockTarget) -> None:
    """Some gateways stream even when stream=false was requested."""
    import httpx

    async def sse_post(self, url, **kwargs):  # type: ignore[no-untyped-def]
        frames = [
            'data: {"choices":[],"usage":{}}',
            'data: {"choices":[{"message":{"content":"streamed answer"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        body = "\n\n".join(frames).encode()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=httpx.Request("POST", url),
        )

    cfg = ProviderConfig(kind="openai-compatible", model="m", base_url="http://x/v1")
    provider = build_provider(cfg)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(httpx.AsyncClient, "post", sse_post)
    try:
        async with httpx.AsyncClient() as client:
            response = await provider.complete(
                [Message(role=Role.USER, content="hi")], client=client
            )
    finally:
        monkeypatch.undo()
    assert response.text == "streamed answer"


async def test_non_json_non_sse_raises(mock_target: MockTarget) -> None:
    import httpx

    async def html_post(self, url, **kwargs):  # type: ignore[no-untyped-def]
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>gateway error</html>",
            request=httpx.Request("POST", url),
        )

    cfg = ProviderConfig(kind="openai-compatible", model="m", base_url="http://x/v1")
    provider = build_provider(cfg)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(httpx.AsyncClient, "post", html_post)
    try:
        async with httpx.AsyncClient() as client:
            with pytest.raises(ProviderError, match="not JSON"):
                await provider.complete([Message(role=Role.USER, content="hi")], client=client)
    finally:
        monkeypatch.undo()


async def test_retries_are_counted_against_request_budget(mock_target: MockTarget) -> None:
    """A retried probe consumes more than one request; budget must reflect it."""
    from aij.engine import run_scan
    from aij.models import Objective, RunConfig, Severity
    from aij.techniques import build_registry

    mock_target.fail_times = 2  # two 500s, then success
    cfg = RunConfig(
        run_id="retry-budget",
        provider=ProviderConfig(
            kind="openai-compatible",
            model="m",
            base_url=mock_target.url,
            api_key="k",
        ),
        concurrency=1,
        retries=3,
    )
    job_obj = Objective(
        id="cyber.reverse-shell",
        category="cyber",
        severity=Severity.HIGH,
        prompt="x",
        success_markers=["socket"],
    )
    outcome = await run_scan(
        config=cfg,
        provider=build_provider(cfg.provider),
        detector=__import__("aij.detectors", fromlist=["HeuristicDetector"]).HeuristicDetector(),
        jobs=[
            __import__("aij.engine.scanner", fromlist=["ScanJob"]).ScanJob(
                objective=job_obj, technique_cls=build_registry().get("direct")
            )
        ],
    )
    assert len(mock_target.requests) == 3
    assert outcome.results[0].turns[0].usage.total_tokens == 46
    # One probe, three wire attempts: the budget must count all three, or
    # --max-requests silently permits 3x the spend it promised.
    assert outcome.requests_sent == 3
