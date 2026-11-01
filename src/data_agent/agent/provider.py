"""Minimal OpenAI-compatible chat-completions client with tool calling."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from .models import ModelProfile


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments_raw: str
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


@dataclass
class ProviderResponse:
    local_request_id: str
    provider_response_id: str | None
    model: str
    created: int | None
    content: str
    reasoning_content: str
    finish_reason: str | None
    tool_calls: list[ToolCall]
    usage: dict[str, Any]
    latency_ms: float
    raw: dict[str, Any]
    request_payload: dict[str, Any]


def _flatten_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        raw_arguments = function.get("arguments")
        arguments: dict[str, Any] = {}
        parse_error = None
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
            raw_arguments = json.dumps(raw_arguments, ensure_ascii=False, sort_keys=True)
        else:
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as error:
                parse_error = str(error)
        calls.append(
            ToolCall(
                call_id=str(item.get("id") or f"call_{uuid.uuid4().hex[:16]}"),
                name=str(function.get("name") or ""),
                arguments_raw=str(raw_arguments or ""),
                arguments=arguments,
                parse_error=parse_error,
            )
        )
    return calls


class ChatProvider:
    """Calls an OpenAI-compatible endpoint and keeps the raw response for audit."""

    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self.endpoint = profile.base_url.rstrip("/") + "/chat/completions"

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
            "max_tokens": self.profile.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.profile.provider == "deepseek":
            # Documented toggle; thinking mode ignores temperature.
            payload["thinking"] = {"type": self.profile.thinking}
        if self.profile.thinking == "disabled":
            payload["temperature"] = self.profile.temperature
        return payload

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        request_index: int,
        extra_headers: dict[str, str] | None = None,
    ) -> ProviderResponse:
        payload = self._payload(messages, tools)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.profile.api_key(),
            "Accept": "application/json",
        }
        headers.update(extra_headers or {})
        local_id = f"{request_index:04d}-{uuid.uuid4().hex[:12]}"
        attempts = self.profile.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.profile.timeout_seconds) as handle:
                    raw_text = handle.read().decode("utf-8")
                latency_ms = (time.perf_counter() - started) * 1000
                raw = json.loads(raw_text)
                return self._response(raw, payload, local_id, latency_ms)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:2000]
                last_error = ProviderError(
                    f"provider HTTP {error.code}: {detail}", status=error.code, body=detail
                )
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error
            except (urllib.error.URLError, socket.timeout, TimeoutError) as error:
                last_error = ProviderError(f"provider connection error: {error}")
            except json.JSONDecodeError as error:
                raise ProviderError(f"provider returned non-JSON body: {error}") from error
            if attempt < attempts - 1:
                time.sleep(min(2 ** (attempt + 1), 8))
        assert last_error is not None
        raise last_error

    def _response(
        self,
        raw: dict[str, Any],
        payload: dict[str, Any],
        local_id: str,
        latency_ms: float,
    ) -> ProviderResponse:
        if "choices" not in raw or not raw["choices"]:
            raise ProviderError(
                "provider response has no choices",
                body=json.dumps(raw, ensure_ascii=False)[:2000],
            )
        choice = raw["choices"][0]
        message = choice.get("message") or {}
        usage = raw.get("usage") or {}
        return ProviderResponse(
            local_request_id=local_id,
            provider_response_id=raw.get("id"),
            model=raw.get("model") or self.profile.model,
            created=raw.get("created"),
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or "",
            finish_reason=choice.get("finish_reason"),
            tool_calls=_flatten_tool_calls(message),
            usage=usage,
            latency_ms=latency_ms,
            raw=raw,
            request_payload=payload,
        )


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Request snapshots are stored without credentials; this is a guard, not a reliance."""
    return {k: v for k, v in payload.items() if k not in {"api_key", "authorization"}}


__all__ = [
    "ChatProvider",
    "ProviderError",
    "ProviderResponse",
    "ToolCall",
    "redact",
]
