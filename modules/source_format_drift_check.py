"""Strict source-format drift checks for Codex session logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared_parsing_utils import coerce_int, list_session_log_files, open_session_log_text, parse_timestamp

KNOWN_ENTRY_TYPES = frozenset({"session_meta", "turn_context", "event_msg", "response_item"})
KNOWN_EVENT_MSG_TYPES = frozenset({"user_message", "token_count"})
KNOWN_RESPONSE_ITEM_TYPES = frozenset({"message", "function_call", "function_call_output"})
REQUIRED_TOKEN_USAGE_KEYS = ("input_tokens", "output_tokens")


@dataclass(frozen=True)
class DriftProblem:
    file_path: str
    line_number: int
    code: str
    details: str

    def render(self) -> str:
        return f"{self.file_path}:{self.line_number} [{self.code}] {self.details}"


class SourceFormatDriftError(RuntimeError):
    def __init__(self, problems: list[DriftProblem], *, max_display: int = 25) -> None:
        self.problems = list(problems)
        self.max_display = max(1, int(max_display))
        super().__init__(self._render_summary())

    def _render_summary(self) -> str:
        counts_by_code: dict[str, int] = {}
        for problem in self.problems:
            counts_by_code[problem.code] = counts_by_code.get(problem.code, 0) + 1

        counts_text = ", ".join(
            f"{code}={count}"
            for code, count in sorted(counts_by_code.items(), key=lambda item: (-item[1], item[0]))
        )
        lines = [
            f"Source format drift detected ({len(self.problems)} issue(s)).",
            f"Issue counts by code: {counts_text}",
            "Examples:",
        ]
        for item in self.problems[: self.max_display]:
            lines.append(f"- {item.render()}")
        hidden = len(self.problems) - self.max_display
        if hidden > 0:
            lines.append(f"- ... and {hidden} more issue(s)")
        return "\n".join(lines)


def _add_problem(
    problems: list[DriftProblem],
    *,
    file_path: Path,
    line_number: int,
    code: str,
    details: str,
) -> None:
    problems.append(
        DriftProblem(
            file_path=str(file_path),
            line_number=int(line_number),
            code=code,
            details=details,
        )
    )


def _require_non_empty_string(
    parent: dict[str, Any],
    key: str,
    *,
    code: str,
    label: str,
    file_path: Path,
    line_number: int,
    problems: list[DriftProblem],
) -> str | None:
    value = parent.get(key)
    if not isinstance(value, str):
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code=code,
            details=f"{label} must be a string",
        )
        return None
    text = value.strip()
    if not text:
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code=code,
            details=f"{label} must not be empty",
        )
        return None
    return text


def _require_dict(
    parent: dict[str, Any],
    key: str,
    *,
    code: str,
    label: str,
    file_path: Path,
    line_number: int,
    problems: list[DriftProblem],
) -> dict[str, Any] | None:
    value = parent.get(key)
    if not isinstance(value, dict):
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code=code,
            details=f"{label} must be an object",
        )
        return None
    return value


def _validate_message_payload(
    payload: dict[str, Any],
    *,
    file_path: Path,
    line_number: int,
    problems: list[DriftProblem],
) -> None:
    _require_non_empty_string(
        payload,
        "role",
        code="message_missing_role",
        label="response_item.payload.role",
        file_path=file_path,
        line_number=line_number,
        problems=problems,
    )
    text_value = payload.get("text")
    content_value = payload.get("content")
    if isinstance(text_value, str) and text_value.strip():
        return
    if isinstance(content_value, list):
        for item in content_value:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                return
    _add_problem(
        problems,
        file_path=file_path,
        line_number=line_number,
        code="message_missing_text_path",
        details="response_item.payload requires non-empty text or content[].text",
    )


def _validate_token_payload(
    payload: dict[str, Any],
    *,
    file_path: Path,
    line_number: int,
    problems: list[DriftProblem],
) -> None:
    info = _require_dict(
        payload,
        "info",
        code="token_usage_path_missing",
        label="event_msg.payload.info",
        file_path=file_path,
        line_number=line_number,
        problems=problems,
    )
    if info is None:
        return

    last_usage = _require_dict(
        info,
        "last_token_usage",
        code="token_usage_path_missing",
        label="event_msg.payload.info.last_token_usage",
        file_path=file_path,
        line_number=line_number,
        problems=problems,
    )
    if last_usage is None:
        return

    for key in REQUIRED_TOKEN_USAGE_KEYS:
        if key not in last_usage:
            _add_problem(
                problems,
                file_path=file_path,
                line_number=line_number,
                code="token_usage_missing_field",
                details=f"event_msg.payload.info.last_token_usage.{key} missing",
            )
            continue
        if coerce_int(last_usage.get(key)) is None:
            _add_problem(
                problems,
                file_path=file_path,
                line_number=line_number,
                code="token_usage_bad_field_type",
                details=f"event_msg.payload.info.last_token_usage.{key} must be int-like",
            )


def _validate_entry(
    entry: Any,
    *,
    file_path: Path,
    line_number: int,
    problems: list[DriftProblem],
) -> None:
    if not isinstance(entry, dict):
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code="entry_not_object",
            details="line payload must be a JSON object",
        )
        return

    entry_type = entry.get("type")
    if not isinstance(entry_type, str) or not entry_type.strip():
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code="missing_entry_type",
            details="top-level type must be a non-empty string",
        )
        return

    timestamp_raw = entry.get("timestamp")
    if not isinstance(timestamp_raw, str) or not timestamp_raw.strip():
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code="missing_timestamp",
            details="top-level timestamp must be a non-empty string",
        )
    elif parse_timestamp(timestamp_raw) is None:
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code="invalid_timestamp",
            details=f"timestamp is not parseable ISO-8601: {timestamp_raw!r}",
        )

    payload = entry.get("payload")
    if not isinstance(payload, dict):
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code="missing_payload",
            details="top-level payload must be an object",
        )
        return

    if entry_type not in KNOWN_ENTRY_TYPES:
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code="unknown_entry_type",
            details=f"unknown top-level type {entry_type!r}",
        )
        return

    if entry_type == "session_meta":
        _require_non_empty_string(
            payload,
            "id",
            code="session_meta_missing_id",
            label="session_meta.payload.id",
            file_path=file_path,
            line_number=line_number,
            problems=problems,
        )
        return

    if entry_type == "turn_context":
        _require_non_empty_string(
            payload,
            "model",
            code="turn_context_missing_model",
            label="turn_context.payload.model",
            file_path=file_path,
            line_number=line_number,
            problems=problems,
        )
        _require_non_empty_string(
            payload,
            "cwd",
            code="turn_context_missing_cwd",
            label="turn_context.payload.cwd",
            file_path=file_path,
            line_number=line_number,
            problems=problems,
        )
        if "sandbox_policy" not in payload:
            _add_problem(
                problems,
                file_path=file_path,
                line_number=line_number,
                code="turn_context_missing_sandbox_policy",
                details="turn_context.payload.sandbox_policy missing",
            )
        _require_non_empty_string(
            payload,
            "approval_policy",
            code="turn_context_missing_approval_policy",
            label="turn_context.payload.approval_policy",
            file_path=file_path,
            line_number=line_number,
            problems=problems,
        )
        return

    payload_type = payload.get("type")
    if not isinstance(payload_type, str) or not payload_type.strip():
        _add_problem(
            problems,
            file_path=file_path,
            line_number=line_number,
            code="missing_payload_type",
            details=f"{entry_type}.payload.type must be a non-empty string",
        )
        return

    if entry_type == "event_msg":
        if payload_type not in KNOWN_EVENT_MSG_TYPES:
            _add_problem(
                problems,
                file_path=file_path,
                line_number=line_number,
                code="unknown_event_msg_type",
                details=f"unknown event_msg payload type {payload_type!r}",
            )
            return
        if payload_type == "user_message":
            _require_non_empty_string(
                payload,
                "message",
                code="user_message_missing_text",
                label="event_msg.payload.message",
                file_path=file_path,
                line_number=line_number,
                problems=problems,
            )
            return
        if payload_type == "token_count":
            _validate_token_payload(
                payload,
                file_path=file_path,
                line_number=line_number,
                problems=problems,
            )
            return

    if entry_type == "response_item":
        if payload_type not in KNOWN_RESPONSE_ITEM_TYPES:
            _add_problem(
                problems,
                file_path=file_path,
                line_number=line_number,
                code="unknown_response_item_type",
                details=f"unknown response_item payload type {payload_type!r}",
            )
            return
        if payload_type == "message":
            _validate_message_payload(
                payload,
                file_path=file_path,
                line_number=line_number,
                problems=problems,
            )
            return
        if payload_type == "function_call":
            _require_non_empty_string(
                payload,
                "name",
                code="function_call_missing_name",
                label="response_item.payload.name",
                file_path=file_path,
                line_number=line_number,
                problems=problems,
            )
            _require_non_empty_string(
                payload,
                "call_id",
                code="function_call_missing_call_id",
                label="response_item.payload.call_id",
                file_path=file_path,
                line_number=line_number,
                problems=problems,
            )
            return
        if payload_type == "function_call_output":
            _require_non_empty_string(
                payload,
                "call_id",
                code="function_call_output_missing_call_id",
                label="response_item.payload.call_id",
                file_path=file_path,
                line_number=line_number,
                problems=problems,
            )
            if "output" not in payload:
                _add_problem(
                    problems,
                    file_path=file_path,
                    line_number=line_number,
                    code="function_call_output_missing_output",
                    details="response_item.payload.output missing",
                )


def collect_source_format_drift(sessions_dir: Path, *, max_problems: int = 50) -> list[DriftProblem]:
    problems: list[DriftProblem] = []
    if max_problems <= 0:
        return problems

    for file_path in list_session_log_files(sessions_dir):
        try:
            with open_session_log_text(file_path) as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if len(problems) >= max_problems:
                        return problems
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        _add_problem(
                            problems,
                            file_path=file_path,
                            line_number=line_number,
                            code="invalid_json",
                            details=f"invalid JSON ({exc.msg})",
                        )
                        continue
                    _validate_entry(
                        entry,
                        file_path=file_path,
                        line_number=line_number,
                        problems=problems,
                    )
        except OSError as exc:
            _add_problem(
                problems,
                file_path=file_path,
                line_number=0,
                code="file_read_error",
                details=str(exc),
            )
            if len(problems) >= max_problems:
                return problems
    return problems


def assert_no_source_format_drift(sessions_dir: Path, *, max_problems: int = 50, max_display: int = 25) -> None:
    problems = collect_source_format_drift(sessions_dir, max_problems=max_problems)
    if problems:
        raise SourceFormatDriftError(problems, max_display=max_display)
