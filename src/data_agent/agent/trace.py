"""Append-only trace, provider receipts and token/cost accounting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_agent.io import sha256_file

from .models import ModelProfile
from .provider import ProviderResponse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceWriter:
    def __init__(self, run_dir: str | Path, profile: ModelProfile):
        self.run_dir = Path(run_dir).resolve()
        self.profile = profile
        self.path = self.run_dir / "trace.jsonl"
        self.usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        }
        self.cost_usd = 0.0
        self.provider_records: list[dict[str, Any]] = []

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"ts": utc_now(), "type": event_type, "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------ provider io
    def record_provider_response(self, response: ProviderResponse, index: int) -> dict[str, Any]:
        requests_dir = self.run_dir / "provider" / "requests"
        responses_dir = self.run_dir / "provider" / "responses"
        request_path = requests_dir / f"{index:04d}.json"
        response_path = responses_dir / f"{response.local_request_id}.json"
        request_path.write_text(
            json.dumps(response.request_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        response_path.write_text(
            json.dumps(response.raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        calls = [
            {
                "id": call.call_id,
                "name": call.name,
                "arguments": call.arguments_raw,
                "arguments_parsed": call.arguments,
                **({"parse_error": call.parse_error} if call.parse_error else {}),
            }
            for call in response.tool_calls
        ]
        usage = self._account(response.usage)
        record = {
            "request_index": index,
            "request_id": response.local_request_id,
            "id": response.provider_response_id or response.local_request_id,
            "model": response.model,
            "created": response.created,
            "finish_reason": response.finish_reason,
            "content": response.content,
            "reasoning_content": response.reasoning_content,
            "tool_calls": calls,
            "usage": response.usage,
            "cost_usd": usage["cost_usd"],
            "latency_ms": round(response.latency_ms, 1),
            "request_file": self._file_record(request_path),
            "response_file": self._file_record(response_path),
            "simulated": False,
        }
        self.provider_records.append(record)
        self.event("provider_response", {k: v for k, v in record.items() if k not in {"request_file", "response_file", "content", "reasoning_content"}})
        return record

    def _file_record(self, path: Path) -> dict[str, str]:
        return {"path": path.relative_to(self.run_dir).as_posix(), "sha256": sha256_file(path)}

    def _account(self, usage: dict[str, Any]) -> dict[str, Any]:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens") or 0)
        hit = int(usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        miss = int(usage.get("prompt_cache_miss_tokens") or max(prompt - hit, 0))
        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += prompt
        self.usage["completion_tokens"] += completion
        self.usage["reasoning_tokens"] += reasoning
        self.usage["cache_hit_tokens"] += hit
        self.usage["cache_miss_tokens"] += miss
        cost = 0.0
        pricing = self.profile.pricing
        if pricing:
            cost = (
                hit / 1_000_000 * pricing.prompt_cache_hit
                + miss / 1_000_000 * pricing.prompt_cache_miss
                + completion / 1_000_000 * pricing.completion
            )
            self.cost_usd += cost
        return {"cost_usd": round(cost, 8)}

    # ---------------------------------------------------------------- summary
    def provider_export(self, run_id: str, source_sha256: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "source_sha256": source_sha256,
            "provider": self.profile.provider,
            "base_url": self.profile.base_url,
            "model_requested": self.profile.model,
            "documented_version": self.profile.documented_version,
            "thinking": self.profile.thinking,
            "request_count": len(self.provider_records),
            "responses": self.provider_records,
            "usage": dict(self.usage),
            "cost": {
                "currency": self.profile.pricing.currency if self.profile.pricing else None,
                "total": round(self.cost_usd, 8),
                "pricing": self.profile.pricing.model_dump() if self.profile.pricing else None,
                "basis": "computed from returned usage and published per-1M-token prices",
            },
            "simulated": False,
        }

    def totals(self) -> dict[str, Any]:
        return {
            "usage": dict(self.usage),
            "cost_usd": round(self.cost_usd, 8),
            "currency": self.profile.pricing.currency if self.profile.pricing else None,
        }


__all__ = ["TraceWriter", "utc_now"]
