"""Deterministic control layer: whitelist, schema validation, risk gates and feedback."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Any

from data_agent.cleaning.strategies import get_strategy

from .provider import ToolCall
from .session import AgentSession
from .tools import ToolContext, ToolError, get
from .trace import TraceWriter


class ValidationFailure(ValueError):
    pass


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _validate_node(schema: dict[str, Any], value: Any, path: str) -> Any:
    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        raise ValidationFailure(f"{path}: expected {expected}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationFailure(f"{path}: value not in allowed enum {schema['enum']}")
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        raise ValidationFailure(f"{path}: string is shorter than {schema['minLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationFailure(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationFailure(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, list):
        items = schema.get("items")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValidationFailure(f"{path}: fewer than {schema['minItems']} items")
        if items:
            value = [_validate_node(items, item, f"{path}[]") for item in value]
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise ValidationFailure(f"{path}: missing required field {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValidationFailure(f"{path}: unexpected fields {sorted(extra)}")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key not in properties:
                result[key] = item
                continue
            result[key] = _validate_node(properties[key], item, f"{path}.{key}" if path else key)
        for key, item_schema in properties.items():
            if key not in result and "default" in item_schema:
                result[key] = copy.deepcopy(item_schema["default"])
        return result
    return value


def validate_arguments(tool_spec: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    schema = tool_spec["function"]["parameters"]
    validated = _validate_node(schema, arguments, "")
    if not isinstance(validated, dict):
        raise ValidationFailure("arguments must be an object")
    return validated


@dataclass(frozen=True)
class RunPolicy:
    max_steps: int = 24
    max_tool_calls: int = 60
    dangerous_policy: str = "deny"  # deny | allow
    max_removal_fraction: float = 0.1

    @property
    def dangerous_allowed(self) -> bool:
        return self.dangerous_policy == "allow"


class ControlLayer:
    def __init__(self, session: AgentSession, trace: TraceWriter, policy: RunPolicy):
        self.session = session
        self.trace = trace
        self.policy = policy
        self.call_count = 0
        self.duplicate_calls: set[str] = set()
        self.records: list[dict[str, Any]] = []

    def execute(self, call: ToolCall, response_id: str) -> dict[str, Any]:
        self.call_count += 1
        started = time.perf_counter()
        before = self.session.data_sha256
        arguments = call.arguments
        payload: dict[str, Any]
        status = "error"
        error_code = None

        if self.call_count > self.policy.max_tool_calls:
            payload = {
                "status": "error",
                "error_code": "tool_budget",
                "message": f"工具调用超过上限 {self.policy.max_tool_calls}，请立即调用 submit_final_report 收尾",
            }
        elif call.parse_error:
            payload = {
                "status": "error",
                "error_code": "invalid_json_arguments",
                "message": (
                    f"工具参数不是合法 JSON：{call.parse_error}。"
                    "若上一次回复因输出上限被截断，请缩短文字（尤其 submit_final_report 的四个字段）后重试"
                ),
            }
        else:
            key = call.name + "|" + json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            if key in self.duplicate_calls:
                payload = {
                    "status": "error",
                    "error_code": "duplicate_call",
                    "message": "相同工具与参数已调用过，禁止原样重复；请更换策略或参数",
                }
            else:
                self.duplicate_calls.add(key)
                try:
                    definition = get(call.name)
                    validated = validate_arguments(definition.spec(), arguments)
                    self._policy_check(definition.name, validated)
                    context = ToolContext(
                        response_id=response_id,
                        tool_call_id=call.call_id,
                        arguments_raw=call.arguments_raw,
                        max_removal_fraction=self.policy.max_removal_fraction,
                        dangerous_allowed=self.policy.dangerous_allowed,
                    )
                    payload = definition.handler(self.session, validated, context)
                    status = "success"
                except ToolError as error:
                    status = "error"
                    error_code = error.code
                    payload = {
                        "status": "error",
                        "error_code": error.code,
                        "message": error.message,
                        **({"details": error.details} if error.details else {}),
                    }
                except ValidationFailure as error:
                    status = "error"
                    error_code = "invalid_arguments"
                    payload = {"status": "error", "error_code": "invalid_arguments", "message": str(error)}
                except Exception as error:  # execution failures must return to the model, not crash
                    status = "error"
                    error_code = "tool_execution_error"
                    payload = {
                        "status": "error",
                        "error_code": "tool_execution_error",
                        "message": f"{type(error).__name__}: {error}",
                    }

        duration = (time.perf_counter() - started) * 1000
        self.records.append(
            {
                "seq": self.call_count,
                "tool": call.name,
                "arguments": call.arguments_raw,
                "status": status,
                "error_code": error_code,
                "duration_ms": round(duration, 1),
                "summary": self._summary(payload),
            }
        )
        self.trace.event(
            "tool_call",
            {
                "seq": self.call_count,
                "response_id": response_id,
                "tool_call_id": call.call_id,
                "tool": call.name,
                "arguments": call.arguments_raw,
                "status": status,
                "error_code": error_code,
                "duration_ms": round(duration, 1),
                "data_sha256_before": before,
                "data_sha256_after": self.session.data_sha256,
                "summary": self._summary(payload),
            },
        )
        return payload

    def _policy_check(self, name: str, args: dict[str, Any]) -> None:
        if name.startswith("clean_") or name == "normalize_types_formats" or name in {"resolve_conflicts", "repair_labels"}:
            issue = {
                "clean_missing": "missing",
                "clean_duplicates": "duplicate",
                "clean_outliers": "outlier",
                "normalize_types_formats": "format",
                "resolve_conflicts": "constraint",
                "repair_labels": "label",
            }[name]
            strategy = get_strategy(issue, args["strategy_id"])
            if strategy.requires_confirmation and not self.policy.dangerous_allowed:
                raise ToolError(
                    "confirmation_required",
                    f"策略 {args['strategy_id']!r} 需要人工授权，本次运行策略为 deny",
                )
            # selection_artifact_id is a hint only: the handler always resolves the
            # artifact from the finding, so a stale id cannot redirect file access.
            for column in args.get("target_columns") or []:
                if column not in self.session.frame.columns:
                    raise ToolError("unknown_column", f"目标列不存在：{column!r}")

    @staticmethod
    def _summary(payload: dict[str, Any]) -> dict[str, Any]:
        keys = ("status", "error_code", "message", "step_index", "rows_after", "cells_changed", "k", "silhouette", "valid_rows", "tool_name")
        return {key: payload[key] for key in keys if key in payload}


__all__ = ["ControlLayer", "RunPolicy", "ValidationFailure", "validate_arguments"]
