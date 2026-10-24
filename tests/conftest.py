"""Shared fixtures: a mock OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest


class MockTarget:
    """A scriptable chat-completions server.

    ``responder(messages)`` returns the assistant text. ``fail_times`` makes the
    first N requests return ``fail_status``, for retry tests.
    """

    def __init__(self) -> None:
        self.responder: Callable[[list[dict[str, Any]]], str] = lambda messages: (
            "I can't help with that."
        )
        # When set, the endpoint returns an OpenAI-style safety block instead of
        # a normal completion: "refusal_field" or "content_filter".
        self.block_mode: str | None = None
        self.requests: list[dict[str, Any]] = []
        self.fail_times = 0
        self.fail_status = 500
        self.url = ""
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> MockTarget:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = {}

                with outer._lock:
                    outer.requests.append(payload)
                    should_fail = outer.fail_times > 0
                    if should_fail:
                        outer.fail_times -= 1

                if should_fail:
                    self._write(
                        outer.fail_status,
                        json.dumps({"error": {"message": "synthetic failure"}}).encode(),
                    )
                    return

                messages = payload.get("messages") or []

                if outer.block_mode == "refusal_field":
                    self._write(
                        200,
                        json.dumps(
                            {
                                "model": payload.get("model", "mock"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": None,
                                            "refusal": "I can't help with that.",
                                        },
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 5,
                                    "completion_tokens": 0,
                                    "total_tokens": 5,
                                },
                            }
                        ).encode(),
                    )
                    return

                if outer.block_mode == "content_filter":
                    self._write(
                        200,
                        json.dumps(
                            {
                                "model": payload.get("model", "mock"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": ""},
                                        "finish_reason": "content_filter",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 5,
                                    "completion_tokens": 0,
                                    "total_tokens": 5,
                                },
                            }
                        ).encode(),
                    )
                    return

                text = outer.responder(messages)
                body = json.dumps(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "model": payload.get("model", "mock"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 34,
                            "total_tokens": 46,
                        },
                    }
                ).encode()
                self._write(200, body)

            def _write(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.url = f"http://{host}:{port}/v1"
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def user_messages(self) -> list[str]:
        out: list[str] = []
        for payload in self.requests:
            for message in payload.get("messages") or []:
                if message.get("role") == "user":
                    out.append(str(message.get("content", "")))
        return out

    def last_user_message(self) -> str:
        messages = self.user_messages
        return messages[-1] if messages else ""


@pytest.fixture
def mock_target() -> MockTarget:
    target = MockTarget().start()
    yield target
    target.stop()
