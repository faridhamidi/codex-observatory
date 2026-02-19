#!/usr/bin/env python3
"""Standalone local mini app for Codex model routing and usage audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from .metrics_audit import build_usage_audit_report
except ImportError:  # pragma: no cover - support direct script execution
    from metrics_audit import build_usage_audit_report


CSV_HEADERS = [
    "timestamp_utc",
    "timestamp_epoch_ms",
    "event_type",
    "thread_id",
    "thread_label",
    "conversation_id",
    "query_id",
    "query_index",
    "model",
    "cwd",
    "sandbox_mode",
    "approval_policy",
    "session_file",
    "message_role",
    "message_text",
    "query_text",
    "tool_name",
    "tool_call_id",
    "tool_start_epoch_ms",
    "tool_end_epoch_ms",
    "tool_duration_ms",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
    "total_tokens_raw",
    "total_tokens_recomputed",
    "total_tokens_delta",
    "reconciliation_status",
    "total_tokens",
]

HTML_NAME = "model_audit_dashboard.html"
AUDIT_JSON_NAME = "model_audit_audit.json"
DATA_DIR_NAME = "data"
DATA_CATALOG_NAME = "catalog.json"
CODEX_GLOBAL_STATE_PATH = Path.home() / ".codex" / ".codex-global-state.json"
DEFAULT_MAX_CSV_ROWS = 0
DEFAULT_MAX_RETENTION_DAYS = 31
MAX_QUERY_TEXT_CHARS = 600
MAX_MESSAGE_TEXT_CHARS = 240

ROUTING_CSV_HEADERS = [
    "timestamp_utc",
    "timestamp_epoch_ms",
    "event_type",
    "thread_id",
    "thread_label",
    "conversation_id",
    "query_id",
    "query_index",
    "model",
    "cwd",
    "sandbox_mode",
    "approval_policy",
    "session_file",
]

QUERY_CSV_HEADERS = [
    "timestamp_utc",
    "timestamp_epoch_ms",
    "event_type",
    "thread_id",
    "thread_label",
    "conversation_id",
    "query_id",
    "query_index",
    "model",
    "cwd",
    "sandbox_mode",
    "approval_policy",
    "session_file",
    "message_role",
    "message_text",
    "query_text",
]

TOKEN_CSV_HEADERS = [
    "timestamp_utc",
    "timestamp_epoch_ms",
    "event_type",
    "thread_id",
    "thread_label",
    "conversation_id",
    "query_id",
    "query_index",
    "model",
    "cwd",
    "sandbox_mode",
    "approval_policy",
    "session_file",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
    "total_tokens_raw",
    "total_tokens_recomputed",
    "total_tokens_delta",
    "reconciliation_status",
    "total_tokens",
]

TOOL_CSV_HEADERS = [
    "timestamp_utc",
    "timestamp_epoch_ms",
    "event_type",
    "thread_id",
    "thread_label",
    "conversation_id",
    "query_id",
    "query_index",
    "model",
    "cwd",
    "sandbox_mode",
    "approval_policy",
    "session_file",
    "tool_name",
    "tool_call_id",
    "tool_start_epoch_ms",
    "tool_end_epoch_ms",
    "tool_duration_ms",
]

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "routing_context": {
        "event_type": "turn_context",
        "file_name": "routing_context.csv",
        "description": "Routing policy snapshots (model/sandbox/approval per turn context).",
        "headers": ROUTING_CSV_HEADERS,
    },
    "query_messages": {
        "event_type": "user_message",
        "file_name": "query_messages.csv",
        "description": "User query messages and normalized query text.",
        "headers": QUERY_CSV_HEADERS,
    },
    "usage_tokens": {
        "event_type": "token_count",
        "file_name": "usage_tokens.csv",
        "description": "Token telemetry and reconciliation metrics per query.",
        "headers": TOKEN_CSV_HEADERS,
    },
    "tool_calls": {
        "event_type": "tool_call",
        "file_name": "tool_calls.csv",
        "description": "Tool invocations and duration timing.",
        "headers": TOOL_CSV_HEADERS,
    },
}


@dataclass
class AuditEventRow:
    timestamp: datetime
    event_type: str
    thread_id: str
    thread_label: str | None = None
    conversation_id: str | None = None
    query_id: str | None = None
    query_index: int | None = None
    model: str = "unknown"
    cwd: str = "unknown"
    sandbox_mode: str = "unknown"
    approval_policy: str = "unknown"
    session_file: str = ""
    message_role: str | None = None
    message_text: str | None = None
    query_text: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_start_epoch_ms: int | None = None
    tool_end_epoch_ms: int | None = None
    tool_duration_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens_raw: int | None = None
    total_tokens_recomputed: int | None = None
    total_tokens_delta: int | None = None
    reconciliation_status: str | None = None
    total_tokens: int | None = None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def as_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_sandbox_mode(policy: Any) -> str:
    if isinstance(policy, dict):
        return str(policy.get("type") or policy.get("mode") or "unknown")
    return "unknown"


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "." in text:
                return int(float(text))
            return int(text)
        except ValueError:
            return None
    return None


def compact_text(value: str | None, max_chars: int) -> str:
    if not value:
        return ""
    normalized = " ".join(value.split())
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3] + "..."


def extract_response_message_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return "\n".join(parts).strip()
    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def parse_tool_duration_ms(output_payload: Any) -> int | None:
    parsed: Any = output_payload
    if isinstance(output_payload, str):
        text = output_payload.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return None

    metadata = parsed.get("metadata")
    if isinstance(metadata, dict):
        dur_ms = coerce_int(metadata.get("duration_ms"))
        if dur_ms is not None:
            return max(0, dur_ms)
        dur_seconds = metadata.get("duration_seconds")
        if isinstance(dur_seconds, (int, float)):
            return max(0, int(dur_seconds * 1000))

    dur_ms = coerce_int(parsed.get("duration_ms"))
    if dur_ms is not None:
        return max(0, dur_ms)
    dur_seconds = parsed.get("duration_seconds")
    if isinstance(dur_seconds, (int, float)):
        return max(0, int(dur_seconds * 1000))
    return None


def load_thread_title_map(global_state_path: Path = CODEX_GLOBAL_STATE_PATH) -> dict[str, str]:
    try:
        raw = global_state_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    section = payload.get("thread-titles")
    if not isinstance(section, dict):
        return {}
    titles = section.get("titles")
    if not isinstance(titles, dict):
        return {}
    mapped: dict[str, str] = {}
    for key, value in titles.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, str):
            continue
        title = value.strip()
        if title:
            mapped[key.strip()] = title
    return mapped


def make_thread_label(first_query_text: str, thread_id: str, cwd: str) -> str:
    lines = [line.strip() for line in first_query_text.splitlines() if line.strip()]
    for line in lines:
        low = line.lower()
        if low.startswith("<environment_context>"):
            continue
        if low.startswith("# context from my ide setup"):
            continue
        if low.startswith("## my request for codex"):
            continue
        cleaned = line.lstrip("-*# ").strip()
        if cleaned:
            return cleaned[:72]

    base = Path(cwd).name if cwd and cwd != "unknown" else "thread"
    suffix = thread_id.split("-")[0][:8] if thread_id else "unknown"
    return f"{base}-{suffix}"[:72]


def extract_event_rows(sessions_dir: Path) -> list[AuditEventRow]:
    rows: list[AuditEventRow] = []
    if not sessions_dir.exists():
        return rows

    thread_title_map = load_thread_title_map()

    for file_path in sorted(sessions_dir.rglob("*.jsonl")):
        file_rows: list[AuditEventRow] = []
        session_file_name = file_path.name
        thread_id = file_path.stem
        context_model = "unknown"
        context_cwd = "unknown"
        context_sandbox_mode = "unknown"
        context_approval_policy = "unknown"
        first_query_text = ""
        first_known_cwd = "unknown"

        query_index = 0
        current_query_id = "unknown"
        current_conversation_id = "unknown"

        open_tools: dict[str, dict[str, Any]] = {}

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts = parse_timestamp(entry.get("timestamp"))
                    if ts is None:
                        continue

                    entry_type = entry.get("type")
                    payload = entry.get("payload")
                    if not isinstance(payload, dict):
                        payload = {}

                    if entry_type == "session_meta":
                        session_id = payload.get("id")
                        if session_id:
                            thread_id = str(session_id)
                        continue

                    if entry_type == "turn_context":
                        model = payload.get("model")
                        if model:
                            context_model = str(model)

                        cwd = payload.get("cwd")
                        if cwd:
                            context_cwd = str(cwd)
                            if first_known_cwd == "unknown":
                                first_known_cwd = context_cwd

                        if payload.get("sandbox_policy") is not None:
                            context_sandbox_mode = normalize_sandbox_mode(payload.get("sandbox_policy"))

                        approval = payload.get("approval_policy")
                        if approval:
                            context_approval_policy = str(approval)

                        file_rows.append(
                            AuditEventRow(
                                timestamp=ts,
                                event_type="turn_context",
                                thread_id=thread_id,
                                conversation_id=current_conversation_id,
                                query_id=current_query_id,
                                query_index=query_index if query_index > 0 else None,
                                model=context_model,
                                cwd=context_cwd,
                                sandbox_mode=context_sandbox_mode,
                                approval_policy=context_approval_policy,
                                session_file=session_file_name,
                            )
                        )
                        continue

                    if entry_type == "event_msg":
                        payload_type = payload.get("type")
                        if payload_type == "user_message":
                            query_index += 1
                            current_query_id = f"{thread_id}:q{query_index}"
                            current_conversation_id = current_query_id

                            message_text = str(payload.get("message") or "").strip()
                            message_text = compact_text(message_text, MAX_QUERY_TEXT_CHARS)
                            if not first_query_text and message_text:
                                first_query_text = message_text

                            file_rows.append(
                                AuditEventRow(
                                    timestamp=ts,
                                    event_type="user_message",
                                    thread_id=thread_id,
                                    conversation_id=current_conversation_id,
                                    query_id=current_query_id,
                                    query_index=query_index,
                                    model=context_model,
                                    cwd=context_cwd,
                                    sandbox_mode=context_sandbox_mode,
                                    approval_policy=context_approval_policy,
                                    session_file=session_file_name,
                                    message_role="user",
                                    message_text=message_text,
                                    query_text=message_text,
                                )
                            )
                            continue

                        if payload_type == "token_count":
                            info = payload.get("info")
                            if not isinstance(info, dict):
                                continue
                            last_usage = info.get("last_token_usage")
                            if not isinstance(last_usage, dict):
                                continue

                            input_tokens = coerce_int(last_usage.get("input_tokens"))
                            output_tokens = coerce_int(last_usage.get("output_tokens"))
                            cached_input_tokens = coerce_int(last_usage.get("cached_input_tokens"))
                            reasoning_output_tokens = coerce_int(last_usage.get("reasoning_output_tokens"))
                            total_tokens_raw = coerce_int(last_usage.get("total_tokens"))

                            # Some compacted/session-summary frames emit a non-zero raw total while all
                            # last-usage components are zero. Those rows are not comparable for raw-vs-recomputed
                            # reconciliation and should not fail A1.
                            is_non_comparable_snapshot = (
                                isinstance(total_tokens_raw, int)
                                and total_tokens_raw > 0
                                and (input_tokens or 0) == 0
                                and (output_tokens or 0) == 0
                                and (cached_input_tokens or 0) == 0
                                and (reasoning_output_tokens or 0) == 0
                            )

                            total_tokens_recomputed: int | None = None
                            if not is_non_comparable_snapshot and (input_tokens is not None or output_tokens is not None):
                                total_tokens_recomputed = (input_tokens or 0) + (output_tokens or 0)

                            total_tokens_delta: int | None = None
                            if total_tokens_raw is not None and total_tokens_recomputed is not None:
                                total_tokens_delta = total_tokens_raw - total_tokens_recomputed

                            if is_non_comparable_snapshot:
                                total_tokens = 0
                            elif total_tokens_recomputed is not None:
                                total_tokens = total_tokens_recomputed
                            else:
                                total_tokens = total_tokens_raw

                            if is_non_comparable_snapshot:
                                reconciliation_status = "non_comparable_snapshot"
                            elif total_tokens_delta is None:
                                reconciliation_status = "insufficient_data"
                            elif total_tokens_delta == 0:
                                reconciliation_status = "match"
                            else:
                                reconciliation_status = "mismatch"

                            file_rows.append(
                                AuditEventRow(
                                    timestamp=ts,
                                    event_type="token_count",
                                    thread_id=thread_id,
                                    conversation_id=current_conversation_id,
                                    query_id=current_query_id,
                                    query_index=query_index if query_index > 0 else None,
                                    model=context_model,
                                    cwd=context_cwd,
                                    sandbox_mode=context_sandbox_mode,
                                    approval_policy=context_approval_policy,
                                    session_file=session_file_name,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    cached_input_tokens=cached_input_tokens,
                                    reasoning_output_tokens=reasoning_output_tokens,
                                    total_tokens_raw=total_tokens_raw,
                                    total_tokens_recomputed=total_tokens_recomputed,
                                    total_tokens_delta=total_tokens_delta,
                                    reconciliation_status=reconciliation_status,
                                    total_tokens=total_tokens,
                                )
                            )
                            continue

                    if entry_type == "response_item":
                        payload_type = payload.get("type")

                        if payload_type == "message":
                            role = str(payload.get("role") or "assistant")
                            if role != "user":
                                continue

                            message_text = extract_response_message_text(payload)
                            if not message_text:
                                continue

                            query_index += 1
                            current_query_id = f"{thread_id}:q{query_index}"
                            current_conversation_id = current_query_id
                            message_text = compact_text(message_text, MAX_QUERY_TEXT_CHARS)
                            if not first_query_text and message_text:
                                first_query_text = message_text

                            file_rows.append(
                                AuditEventRow(
                                    timestamp=ts,
                                    event_type="user_message",
                                    thread_id=thread_id,
                                    conversation_id=current_conversation_id,
                                    query_id=current_query_id,
                                    query_index=query_index if query_index > 0 else None,
                                    model=context_model,
                                    cwd=context_cwd,
                                    sandbox_mode=context_sandbox_mode,
                                    approval_policy=context_approval_policy,
                                    session_file=session_file_name,
                                    message_role=role,
                                    message_text=message_text,
                                    query_text=message_text,
                                )
                            )
                            continue

                        if payload_type == "function_call":
                            call_id = str(payload.get("call_id") or "").strip()
                            if not call_id:
                                continue

                            open_tools[call_id] = {
                                "tool_name": str(payload.get("name") or "unknown_tool"),
                                "start_ts": ts,
                                "thread_id": thread_id,
                                "conversation_id": current_conversation_id,
                                "query_id": current_query_id,
                                "query_index": query_index if query_index > 0 else None,
                                "model": context_model,
                                "cwd": context_cwd,
                                "sandbox_mode": context_sandbox_mode,
                                "approval_policy": context_approval_policy,
                                "session_file": session_file_name,
                            }
                            continue

                        if payload_type == "function_call_output":
                            call_id = str(payload.get("call_id") or "").strip()
                            if not call_id:
                                continue

                            start_info = open_tools.pop(call_id, None)
                            if not start_info:
                                continue

                            start_ts = start_info["start_ts"]
                            start_ms = int(start_ts.timestamp() * 1000)
                            end_ms = int(ts.timestamp() * 1000)
                            duration_ms = parse_tool_duration_ms(payload.get("output"))
                            if duration_ms is None:
                                duration_ms = max(0, end_ms - start_ms)

                            file_rows.append(
                                AuditEventRow(
                                    timestamp=ts,
                                    event_type="tool_call",
                                    thread_id=str(start_info["thread_id"]),
                                    conversation_id=str(start_info["conversation_id"]),
                                    query_id=str(start_info["query_id"]),
                                    query_index=start_info.get("query_index"),
                                    model=str(start_info["model"]),
                                    cwd=str(start_info["cwd"]),
                                    sandbox_mode=str(start_info["sandbox_mode"]),
                                    approval_policy=str(start_info["approval_policy"]),
                                    session_file=str(start_info["session_file"]),
                                    tool_name=str(start_info["tool_name"]),
                                    tool_call_id=call_id,
                                    tool_start_epoch_ms=start_ms,
                                    tool_end_epoch_ms=end_ms,
                                    tool_duration_ms=duration_ms,
                                )
                            )
                            continue

        except OSError:
            continue

        for call_id, start_info in open_tools.items():
            start_ts = start_info["start_ts"]
            start_ms = int(start_ts.timestamp() * 1000)
            file_rows.append(
                AuditEventRow(
                    timestamp=start_ts,
                    event_type="tool_call",
                    thread_id=str(start_info["thread_id"]),
                    conversation_id=str(start_info["conversation_id"]),
                    query_id=str(start_info["query_id"]),
                    query_index=start_info.get("query_index"),
                    model=str(start_info["model"]),
                    cwd=str(start_info["cwd"]),
                    sandbox_mode=str(start_info["sandbox_mode"]),
                    approval_policy=str(start_info["approval_policy"]),
                    session_file=str(start_info["session_file"]),
                    tool_name=str(start_info["tool_name"]),
                    tool_call_id=call_id,
                    tool_start_epoch_ms=start_ms,
                )
            )

        thread_label = thread_title_map.get(thread_id, "").strip()
        if not thread_label:
            thread_label = make_thread_label(
                first_query_text=first_query_text,
                thread_id=thread_id,
                cwd=first_known_cwd,
            )
        for row in file_rows:
            row.thread_label = thread_label
        rows.extend(file_rows)

    rows.sort(key=lambda item: item.timestamp)
    return rows


def _csv_value(value: Any) -> str | int:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return value
    return str(value)


def _row_to_csv_dict(row: AuditEventRow) -> dict[str, str | int]:
    return {
        "timestamp_utc": as_utc_iso(row.timestamp),
        "timestamp_epoch_ms": int(row.timestamp.timestamp() * 1000),
        "event_type": row.event_type,
        "thread_id": _csv_value(row.thread_id),
        "thread_label": _csv_value(row.thread_label),
        "conversation_id": _csv_value(row.conversation_id),
        "query_id": _csv_value(row.query_id),
        "query_index": _csv_value(row.query_index),
        "model": _csv_value(row.model),
        "cwd": _csv_value(row.cwd),
        "sandbox_mode": _csv_value(row.sandbox_mode),
        "approval_policy": _csv_value(row.approval_policy),
        "session_file": _csv_value(row.session_file),
        "message_role": _csv_value(row.message_role),
        "message_text": _csv_value(row.message_text),
        "query_text": _csv_value(row.query_text),
        "tool_name": _csv_value(row.tool_name),
        "tool_call_id": _csv_value(row.tool_call_id),
        "tool_start_epoch_ms": _csv_value(row.tool_start_epoch_ms),
        "tool_end_epoch_ms": _csv_value(row.tool_end_epoch_ms),
        "tool_duration_ms": _csv_value(row.tool_duration_ms),
        "input_tokens": _csv_value(row.input_tokens),
        "output_tokens": _csv_value(row.output_tokens),
        "cached_input_tokens": _csv_value(row.cached_input_tokens),
        "reasoning_output_tokens": _csv_value(row.reasoning_output_tokens),
        "total_tokens_raw": _csv_value(row.total_tokens_raw),
        "total_tokens_recomputed": _csv_value(row.total_tokens_recomputed),
        "total_tokens_delta": _csv_value(row.total_tokens_delta),
        "reconciliation_status": _csv_value(row.reconciliation_status),
        "total_tokens": _csv_value(row.total_tokens),
    }


def trim_rows_for_csv_budget(
    rows: list[AuditEventRow],
    *,
    max_retention_days: int,
    max_rows: int,
) -> tuple[list[AuditEventRow], dict[str, Any]]:
    original_count = len(rows)
    trimmed_by_age_limit = 0
    trimmed_by_row_limit = 0
    retention_cutoff_utc = ""
    latest_event_utc = ""

    effective_rows = rows
    latest_ts: datetime | None = None
    if effective_rows:
        latest_ts = effective_rows[-1].timestamp
        latest_event_utc = as_utc_iso(latest_ts)

    if max_retention_days > 0 and latest_ts is not None:
        cutoff_ts = latest_ts - timedelta(days=max_retention_days)
        retention_cutoff_utc = as_utc_iso(cutoff_ts)
        start_idx = 0
        while start_idx < len(effective_rows) and effective_rows[start_idx].timestamp < cutoff_ts:
            start_idx += 1
        trimmed_by_age_limit = start_idx
        effective_rows = effective_rows[start_idx:]

    if max_rows > 0 and len(effective_rows) > max_rows:
        trimmed_by_row_limit = len(effective_rows) - max_rows
        effective_rows = effective_rows[-max_rows:]

    return (
        effective_rows,
        {
            "original_rows": original_count,
            "rows_after_limits": len(effective_rows),
            "trimmed_by_age_limit": trimmed_by_age_limit,
            "trimmed_by_row_limit": trimmed_by_row_limit,
            "max_retention_days": max_retention_days,
            "latest_event_utc": latest_event_utc,
            "retention_cutoff_utc": retention_cutoff_utc,
        },
    )


def write_csv_atomic(
    csv_path: Path,
    rows: list[AuditEventRow],
    *,
    headers: list[str] | None = None,
) -> None:
    fieldnames = headers or CSV_HEADERS
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(csv_path.parent),
        prefix=".tmp_model_audit_",
        suffix=".csv",
    ) as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_payload = _row_to_csv_dict(row)
            writer.writerow({name: row_payload.get(name, "") for name in fieldnames})
        temp_path = Path(temp_file.name)

    temp_path.replace(csv_path)


def write_dataset_csvs_atomic(data_dir: Path, rows: list[AuditEventRow]) -> dict[str, dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_meta: dict[str, dict[str, Any]] = {}
    for dataset_key, spec in DATASET_SPECS.items():
        event_type = str(spec["event_type"])
        headers = list(spec["headers"])
        file_name = str(spec["file_name"])
        csv_path = data_dir / file_name
        dataset_rows = [row for row in rows if row.event_type == event_type]
        write_csv_atomic(csv_path, dataset_rows, headers=headers)
        dataset_meta[dataset_key] = {
            "key": dataset_key,
            "event_type": event_type,
            "file_name": file_name,
            "description": str(spec["description"]),
            "required_fields": headers,
            "row_count": len(dataset_rows),
            "csv_path": str(csv_path),
            "relative_path": f"/{DATA_DIR_NAME}/{file_name}",
            "csv_bytes": csv_path.stat().st_size if csv_path.exists() else 0,
        }
    return dataset_meta


class AuditAppContext:
    def __init__(
        self,
        sessions_dir: Path,
        reports_dir: Path,
        *,
        max_retention_days: int = DEFAULT_MAX_RETENTION_DAYS,
        max_csv_rows: int = DEFAULT_MAX_CSV_ROWS,
    ) -> None:
        self.sessions_dir = sessions_dir
        self.reports_dir = reports_dir
        self.html_path = reports_dir / HTML_NAME
        self.data_dir = reports_dir / DATA_DIR_NAME
        self.data_catalog_path = self.data_dir / DATA_CATALOG_NAME
        self.audit_json_path = reports_dir / AUDIT_JSON_NAME
        self.max_retention_days = max_retention_days
        self.max_csv_rows = max_csv_rows
        self.last_refresh_meta: dict[str, Any] | None = None

    def refresh_csv(self) -> dict[str, Any]:
        generated_at = datetime.now(timezone.utc)
        generated_at_utc = as_utc_iso(generated_at)
        rows = extract_event_rows(self.sessions_dir)
        rows, trim_stats = trim_rows_for_csv_budget(
            rows,
            max_retention_days=self.max_retention_days,
            max_rows=self.max_csv_rows,
        )
        dataset_meta = write_dataset_csvs_atomic(self.data_dir, rows)
        total_data_bytes = sum(int(item.get("csv_bytes") or 0) for item in dataset_meta.values())

        latest_event_epoch_ms = 0
        latest_event_utc = str(trim_stats.get("latest_event_utc", ""))
        if latest_event_utc:
            latest_dt = parse_timestamp(latest_event_utc)
            if latest_dt is not None:
                latest_event_epoch_ms = int(latest_dt.timestamp() * 1000)

        audit_report = build_usage_audit_report(rows=rows, generated_at_utc=generated_at_utc)
        self.audit_json_path.write_text(
            json.dumps(audit_report, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        data_catalog = {
            "status": "ok",
            "generated_at_utc": generated_at_utc,
            "latest_event_utc": latest_event_utc,
            "latest_event_epoch_ms": latest_event_epoch_ms,
            "source_sessions_dir": str(self.sessions_dir),
            "max_retention_days": self.max_retention_days,
            "retention_cutoff_utc": trim_stats["retention_cutoff_utc"],
            "datasets": list(dataset_meta.values()),
        }
        self.data_catalog_path.write_text(
            json.dumps(data_catalog, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        message: str | None = None
        if not self.sessions_dir.exists():
            message = f"Sessions directory does not exist: {self.sessions_dir}"
        elif not rows:
            message = "No supported events found in sessions logs."

        meta = {
            "status": "ok",
            "rows_written": len(rows),
            "rows_source_total": trim_stats["original_rows"],
            "rows_trimmed_total": trim_stats["trimmed_by_age_limit"] + trim_stats["trimmed_by_row_limit"],
            "rows_trimmed_by_age_limit": trim_stats["trimmed_by_age_limit"],
            "rows_trimmed_by_row_limit": trim_stats["trimmed_by_row_limit"],
            "max_retention_days": self.max_retention_days,
            "retention_cutoff_utc": trim_stats["retention_cutoff_utc"],
            "latest_event_utc": trim_stats["latest_event_utc"],
            "max_csv_rows": self.max_csv_rows,
            "estimated_csv_bytes": total_data_bytes,
            "generated_at_utc": generated_at_utc,
            "source_sessions_dir": str(self.sessions_dir),
            "data_dir": str(self.data_dir),
            "data_catalog_path": str(self.data_catalog_path),
            "audit_json_path": str(self.audit_json_path),
            "audit_status": str(audit_report.get("status", "unknown")),
            "metrics_version": str(audit_report.get("metrics_version", "unknown")),
            "latest_event_epoch_ms": latest_event_epoch_ms,
            "dataset_rows": {key: int(value["row_count"]) for key, value in dataset_meta.items()},
        }
        if message:
            meta["message"] = message
        self.last_refresh_meta = meta
        return meta

    def load_dashboard_html(self) -> str:
        text = self.html_path.read_text(encoding="utf-8")
        refresh_time = ""
        if self.last_refresh_meta:
            refresh_time = str(self.last_refresh_meta.get("generated_at_utc", ""))
        text = text.replace("__SOURCE_SESSIONS_DIR__", str(self.sessions_dir))
        text = text.replace("__DATA_BASE_PATH__", f"/{DATA_DIR_NAME}")
        text = text.replace("__AUDIT_JSON_RELATIVE_PATH__", "/audit.json")
        text = text.replace("__LAST_REFRESH_UTC__", refresh_time)
        return text

    def ensure_data(self) -> dict[str, Any]:
        has_csv = False
        for spec in DATASET_SPECS.values():
            file_name = str(spec["file_name"])
            if (self.data_dir / file_name).exists():
                has_csv = True
                break
        if has_csv and self.data_catalog_path.exists():
            return self.last_refresh_meta or {"status": "ok"}
        return self.refresh_csv()

    def read_dataset_csv(
        self,
        dataset_file_name: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> bytes:
        csv_path = self.data_dir / dataset_file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV missing: {csv_path}")
        if start_ms is None and end_ms is None:
            return csv_path.read_bytes()

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            output = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                delete=False,
                dir=str(csv_path.parent),
                prefix=".tmp_model_audit_window_",
                suffix=".csv",
            )
            temp_path = Path(output.name)
            try:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    ts_text = row.get("timestamp_epoch_ms")
                    if not ts_text:
                        continue
                    try:
                        ts = int(ts_text)
                    except ValueError:
                        continue
                    if start_ms is not None and ts < start_ms:
                        continue
                    if end_ms is not None and ts > end_ms:
                        continue
                    writer.writerow(row)
            finally:
                output.close()

        try:
            payload = temp_path.read_bytes()
        finally:
            temp_path.unlink(missing_ok=True)
        return payload


class AuditDashboardHandler(BaseHTTPRequestHandler):
    server_version = "ModelAuditLocalApp/2.0"

    @property
    def app(self) -> AuditAppContext:
        return self.server.app_context  # type: ignore[attr-defined]

    def _send_bytes(
        self,
        code: int,
        body: bytes,
        content_type: str,
        *,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self._send_bytes(code, body, "application/json; charset=utf-8")

    def _send_not_found(self) -> None:
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"status": "error", "message": "Not found"},
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        sys.stdout.write(
            "%s - - [%s] %s\n"
            % (
                self.address_string(),
                self.log_date_time_string(),
                format % args,
            )
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            try:
                html = self.app.load_dashboard_html().encode("utf-8")
            except FileNotFoundError:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "status": "error",
                        "message": f"Dashboard template missing: {self.app.html_path}",
                    },
                )
                return
            self._send_bytes(HTTPStatus.OK, html, "text/html; charset=utf-8")
            return

        if parsed.path.startswith(f"/{DATA_DIR_NAME}/"):
            meta = self.app.ensure_data()
            if meta.get("status") != "ok":
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, meta)
                return

            relative = parsed.path[len(f"/{DATA_DIR_NAME}/") :]
            if relative == DATA_CATALOG_NAME:
                body = self.app.data_catalog_path.read_bytes()
                self._send_bytes(HTTPStatus.OK, body, "application/json; charset=utf-8")
                return

            allowed_files = {str(spec["file_name"]) for spec in DATASET_SPECS.values()}
            if relative not in allowed_files:
                self._send_not_found()
                return

            qs = parse_qs(parsed.query)

            def _query_int(name: str) -> int | None:
                raw = qs.get(name, [""])[0].strip()
                if not raw:
                    return None
                try:
                    return int(raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid query parameter '{name}': {raw}") from exc

            try:
                start_ms = _query_int("start_ms")
                end_ms = _query_int("end_ms")
                body = self.app.read_dataset_csv(relative, start_ms=start_ms, end_ms=end_ms)
            except FileNotFoundError:
                self._send_not_found()
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(exc)})
                return

            self._send_bytes(HTTPStatus.OK, body, "text/csv; charset=utf-8")
            return

        if parsed.path == "/audit.json":
            if not self.app.audit_json_path.exists():
                meta = self.app.ensure_data()
                if meta.get("status") != "ok":
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, meta)
                    return
            body = self.app.audit_json_path.read_bytes()
            self._send_bytes(HTTPStatus.OK, body, "application/json; charset=utf-8")
            return

        self._send_not_found()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/refresh":
            try:
                meta = self.app.refresh_csv()
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"status": "error", "message": f"Refresh failed: {exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, meta)
            return

        self._send_not_found()


def parse_args() -> argparse.Namespace:
    default_reports_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run local model audit mini app (typed CSV-backed dashboard).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--sessions-dir",
        default=str(Path.home() / ".codex" / "sessions"),
        help="Codex sessions directory (default: ~/.codex/sessions)",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(default_reports_dir),
        help=f"Reports directory containing {HTML_NAME}, {AUDIT_JSON_NAME}, and {DATA_DIR_NAME}/",
    )
    parser.add_argument(
        "--max-retention-days",
        type=int,
        default=DEFAULT_MAX_RETENTION_DAYS,
        help=f"Maximum age of events in days retained in typed CSV datasets (default: {DEFAULT_MAX_RETENTION_DAYS})",
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=DEFAULT_MAX_CSV_ROWS,
        help=(
            "Optional hard row cap after age retention (default: 0 = disabled, no row truncation). "
            f"Current default: {DEFAULT_MAX_CSV_ROWS}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sessions_dir = Path(args.sessions_dir).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()

    app_context = AuditAppContext(
        sessions_dir=sessions_dir,
        reports_dir=reports_dir,
        max_retention_days=max(0, args.max_retention_days),
        max_csv_rows=max(0, args.max_csv_rows),
    )
    if not app_context.html_path.exists():
        raise SystemExit(f"Missing dashboard template: {app_context.html_path}")

    startup_meta = app_context.refresh_csv()
    print(
        "[model-audit] startup refresh: "
        f"rows={startup_meta['rows_written']} "
        f"trimmed={startup_meta['rows_trimmed_total']} "
        f"retention_days={startup_meta['max_retention_days']} "
        f"estimated_csv_bytes={startup_meta['estimated_csv_bytes']} "
        f"generated_at={startup_meta['generated_at_utc']}"
    )
    print(f"[model-audit] source sessions: {sessions_dir}")
    print(f"[model-audit] data dir: {app_context.data_dir}")

    server = ThreadingHTTPServer((args.host, args.port), AuditDashboardHandler)
    server.app_context = app_context  # type: ignore[attr-defined]

    print(f"[model-audit] serving at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
