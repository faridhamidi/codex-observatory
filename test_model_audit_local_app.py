from __future__ import annotations

import csv
import json
from pathlib import Path

from model_audit_mini_app import model_audit_local_app as app_module
from model_audit_mini_app.metrics_audit import build_usage_audit_report
from model_audit_mini_app.model_audit_local_app import (
    AUDIT_JSON_NAME,
    AuditAppContext,
    CSV_HEADERS,
    DATA_CATALOG_NAME,
    DATA_DIR_NAME,
    DATASET_SPECS,
    extract_event_rows,
    load_thread_title_map,
    parse_timestamp,
    write_csv_atomic,
)


def test_parse_timestamp_accepts_zulu() -> None:
    ts = parse_timestamp("2026-02-17T05:00:00Z")
    assert ts is not None
    assert ts.isoformat().startswith("2026-02-17T05:00:00")


def test_extract_event_rows_turn_token_and_tool_attribution(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:00:00Z","payload":{"id":"thread-123"}}',
                '{"type":"turn_context","timestamp":"2026-02-17T01:00:01Z","payload":{"model":"gpt-5.3-codex","cwd":"/repo","sandbox_policy":{"mode":"read-only"},"approval_policy":"never"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:02Z","payload":{"type":"user_message","message":"analyze this"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:03Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":100,"cached_input_tokens":40,"output_tokens":20,"reasoning_output_tokens":10,"total_tokens":120}}}}',
                '{"type":"response_item","timestamp":"2026-02-17T01:00:04Z","payload":{"type":"function_call","name":"shell","call_id":"call_1"}}',
                '{"type":"response_item","timestamp":"2026-02-17T01:00:05Z","payload":{"type":"function_call_output","call_id":"call_1","output":"{\\"metadata\\":{\\"duration_seconds\\":0.5}}"}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    assert len(rows) == 4

    turn = next(r for r in rows if r.event_type == "turn_context")
    user_msg = next(r for r in rows if r.event_type == "user_message")
    token = next(r for r in rows if r.event_type == "token_count")
    tool = next(r for r in rows if r.event_type == "tool_call")

    assert turn.thread_id == "thread-123"
    assert turn.model == "gpt-5.3-codex"
    assert turn.cwd == "/repo"
    assert turn.sandbox_mode == "read-only"
    assert turn.approval_policy == "never"

    assert user_msg.query_id == "thread-123:q1"
    assert user_msg.conversation_id == "thread-123:q1"
    assert user_msg.query_text == "analyze this"

    assert token.query_id == "thread-123:q1"
    assert token.input_tokens == 100
    assert token.cached_input_tokens == 40
    assert token.output_tokens == 20
    assert token.reasoning_output_tokens == 10
    assert token.total_tokens_raw == 120
    assert token.total_tokens_recomputed == 120
    assert token.total_tokens_delta == 0
    assert token.reconciliation_status == "match"
    assert token.total_tokens == 120

    assert tool.tool_name == "shell"
    assert tool.tool_call_id == "call_1"
    assert tool.tool_duration_ms == 500


def test_extract_event_rows_fallback_thread_and_malformed_lines(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout-file-stem.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"event_msg","timestamp":"2026-02-17T01:02:03Z","payload":{"type":"user_message","message":"q1"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:02:04Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":10,"output_tokens":5}}}}',
                '{bad-json',
                '{"type":"event_msg","timestamp":"bad-ts","payload":{"type":"user_message","message":"ignored"}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    assert len(rows) == 2

    user = next(r for r in rows if r.event_type == "user_message")
    token = next(r for r in rows if r.event_type == "token_count")

    assert user.thread_id == "rollout-file-stem"
    assert user.query_id == "rollout-file-stem:q1"
    assert token.total_tokens_raw is None
    assert token.total_tokens_recomputed == 15
    assert token.reconciliation_status == "insufficient_data"
    assert token.total_tokens == 15


def test_extract_event_rows_marks_snapshot_totals_non_comparable(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:00:00Z","payload":{"id":"thread-snapshot"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:01Z","payload":{"type":"user_message","message":"q1"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:02Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"reasoning_output_tokens":0,"total_tokens":155555}}}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    token = next(r for r in rows if r.event_type == "token_count")

    assert token.total_tokens_raw == 155555
    assert token.total_tokens_recomputed is None
    assert token.total_tokens_delta is None
    assert token.reconciliation_status == "non_comparable_snapshot"
    assert token.total_tokens == 0

    report = build_usage_audit_report(rows, generated_at_utc="2026-02-17T01:05:00Z")
    assert report["status"] == "pass"
    assert report["summary"]["non_comparable_snapshot_rows"] == 1
    assert report["summary"]["rows_with_both_totals"] == 0


def test_extract_event_rows_skips_non_user_response_messages(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:00:00Z","payload":{"id":"thread-resp"}}',
                '{"type":"response_item","timestamp":"2026-02-17T01:00:01Z","payload":{"type":"message","role":"assistant","text":"assistant text should be dropped"}}',
                '{"type":"response_item","timestamp":"2026-02-17T01:00:02Z","payload":{"type":"message","role":"user","text":"user text should be kept"}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    user_rows = [r for r in rows if r.event_type == "user_message"]
    assert len(user_rows) == 1
    assert user_rows[0].message_role == "user"
    assert user_rows[0].query_text == "user text should be kept"


def test_extract_event_rows_two_queries_deterministic_attribution(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:00:00Z","payload":{"id":"thread-x"}}',
                '{"type":"turn_context","timestamp":"2026-02-17T01:00:01Z","payload":{"model":"gpt-5.3-codex","cwd":"/repo-a","sandbox_policy":{"type":"workspace-write"},"approval_policy":"on-request"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:02Z","payload":{"type":"user_message","message":"first query"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:03Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":50,"output_tokens":10,"total_tokens":60}}}}',
                '{"type":"turn_context","timestamp":"2026-02-17T01:00:04Z","payload":{"model":"gpt-5.3-codex","cwd":"/repo-b","sandbox_policy":{"type":"workspace-write"},"approval_policy":"on-request"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:05Z","payload":{"type":"user_message","message":"second query"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:06Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":30,"output_tokens":10,"total_tokens":40}}}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    token_rows = [r for r in rows if r.event_type == "token_count"]

    assert len(token_rows) == 2
    by_query = {r.query_id: r.total_tokens for r in token_rows}
    assert by_query["thread-x:q1"] == 60
    assert by_query["thread-x:q2"] == 40

    by_workspace = {}
    for row in token_rows:
        by_workspace[row.cwd] = by_workspace.get(row.cwd, 0) + (row.total_tokens or 0)
    assert by_workspace["/repo-a"] == 60
    assert by_workspace["/repo-b"] == 40


def test_load_thread_title_map_reads_titles(tmp_path: Path) -> None:
    path = tmp_path / "global-state.json"
    path.write_text(
        '{"thread-titles":{"titles":{"abc":"Friendly Thread","x":""},"order":["abc"]}}',
        encoding="utf-8",
    )
    titles = load_thread_title_map(path)
    assert titles == {"abc": "Friendly Thread"}


def test_extract_event_rows_prefers_global_thread_title(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "load_thread_title_map", lambda: {"thread-123": "Readable Thread"})

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:00:00Z","payload":{"id":"thread-123"}}',
                '{"type":"turn_context","timestamp":"2026-02-17T01:00:01Z","payload":{"model":"gpt-5.3-codex","cwd":"/repo","sandbox_policy":{"mode":"read-only"},"approval_policy":"never"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:02Z","payload":{"type":"user_message","message":"analyze this"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:03Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":10,"output_tokens":5}}}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    assert rows
    assert all(r.thread_label == "Readable Thread" for r in rows)


def test_write_csv_atomic_outputs_strict_schema(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:02:00Z","payload":{"id":"thread-csv"}}',
                '{"type":"turn_context","timestamp":"2026-02-17T01:02:03Z","payload":{"model":"gpt-5.3-codex","cwd":"/repo","sandbox_policy":{"type":"workspace-write"},"approval_policy":"on-request"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:02:04Z","payload":{"type":"user_message","message":"hello"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:02:05Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":11,"output_tokens":7}}}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    assert rows

    csv_path = tmp_path / "usage_tokens.csv"
    write_csv_atomic(csv_path, rows)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_HEADERS
        data = list(reader)

    assert len(data) == len(rows)
    token_row = next(r for r in data if r["event_type"] == "token_count")
    assert token_row["thread_id"] == "thread-csv"
    assert token_row["input_tokens"] == "11"
    assert token_row["output_tokens"] == "7"
    assert token_row["total_tokens_recomputed"] == "18"
    assert token_row["reconciliation_status"] == "insufficient_data"
    assert token_row["total_tokens"] == "18"
    assert token_row["tool_name"] == ""


def test_usage_audit_report_flags_total_reconciliation_mismatch(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "mismatch.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:00:00Z","payload":{"id":"thread-mismatch"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:02Z","payload":{"type":"user_message","message":"q1"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:03Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":100,"output_tokens":20,"cached_input_tokens":30,"reasoning_output_tokens":10,"total_tokens":121}}}}',
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_event_rows(sessions)
    report = build_usage_audit_report(rows, generated_at_utc="2026-02-17T01:05:00Z")

    assert report["status"] == "fail"
    assert report["metrics_version"] == "v1.0"
    assert report["summary"]["total_tokens_raw_sum"] == 121
    assert report["summary"]["total_tokens_recomputed_sum"] == 120
    assert report["summary"]["total_tokens_delta_raw_minus_recomputed"] == 1
    assert report["summary"]["row_mismatch_count"] == 1
    assert "A1_total_tokens_reconcile" in report["failed_check_ids"]
    assert report["action_items"]
    assert report["action_items"][0]["check_id"] == "A1_total_tokens_reconcile"
    a1 = next(c for c in report["checks"] if c["check_id"] == "A1_total_tokens_reconcile")
    assert a1["status"] == "fail"


def test_refresh_writes_audit_json(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    log_file = sessions / "rollout.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T01:00:00Z","payload":{"id":"thread-1"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:02Z","payload":{"type":"user_message","message":"q1"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T01:00:03Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":50,"output_tokens":5,"total_tokens":55}}}}',
            ]
        ),
        encoding="utf-8",
    )

    reports = tmp_path / "reports"
    reports.mkdir()
    html_path = reports / "model_audit_dashboard.html"
    html_path.write_text(
        "<html>__DATA_BASE_PATH__ __AUDIT_JSON_RELATIVE_PATH__</html>",
        encoding="utf-8",
    )

    app_context = AuditAppContext(sessions_dir=sessions, reports_dir=reports)
    meta = app_context.refresh_csv()

    assert meta["status"] == "ok"
    assert meta["metrics_version"] == "v1.0"
    audit_path = reports / AUDIT_JSON_NAME
    assert audit_path.exists()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["metrics_version"] == "v1.0"
    assert payload["summary"]["total_tokens_reported"] == 55
    assert payload["status"] == "pass"
    assert payload["action_items"] == []

    data_dir = reports / DATA_DIR_NAME
    assert data_dir.exists()
    for spec in DATASET_SPECS.values():
        assert (data_dir / str(spec["file_name"])).exists()

    catalog = json.loads((data_dir / DATA_CATALOG_NAME).read_text(encoding="utf-8"))
    assert catalog["status"] == "ok"
    assert len(catalog["datasets"]) == len(DATASET_SPECS)
    usage_dataset = next(d for d in catalog["datasets"] if d["key"] == "usage_tokens")
    assert usage_dataset["row_count"] == 1


def test_refresh_keeps_only_latest_31_days(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    old_file = sessions / "old.jsonl"
    old_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-01-01T00:00:00Z","payload":{"id":"thread-old"}}',
                '{"type":"event_msg","timestamp":"2026-01-01T00:00:01Z","payload":{"type":"user_message","message":"old q"}}',
                '{"type":"event_msg","timestamp":"2026-01-01T00:00:02Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":10,"output_tokens":1,"total_tokens":11}}}}',
            ]
        ),
        encoding="utf-8",
    )

    new_file = sessions / "new.jsonl"
    new_file.write_text(
        "\n".join(
            [
                '{"type":"session_meta","timestamp":"2026-02-17T00:00:00Z","payload":{"id":"thread-new"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T00:00:01Z","payload":{"type":"user_message","message":"new q"}}',
                '{"type":"event_msg","timestamp":"2026-02-17T00:00:02Z","payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":20,"output_tokens":2,"total_tokens":22}}}}',
            ]
        ),
        encoding="utf-8",
    )

    reports = tmp_path / "reports"
    reports.mkdir()
    html_path = reports / "model_audit_dashboard.html"
    html_path.write_text(
        "<html>__DATA_BASE_PATH__ __AUDIT_JSON_RELATIVE_PATH__</html>",
        encoding="utf-8",
    )

    app_context = AuditAppContext(sessions_dir=sessions, reports_dir=reports)
    meta = app_context.refresh_csv()

    assert meta["max_retention_days"] == 31
    assert meta["rows_trimmed_by_age_limit"] >= 2

    data_dir = reports / DATA_DIR_NAME
    usage_csv_path = data_dir / str(DATASET_SPECS["usage_tokens"]["file_name"])
    query_csv_path = data_dir / str(DATASET_SPECS["query_messages"]["file_name"])

    with usage_csv_path.open("r", encoding="utf-8", newline="") as handle:
        usage_data = list(csv.DictReader(handle))
    with query_csv_path.open("r", encoding="utf-8", newline="") as handle:
        query_data = list(csv.DictReader(handle))

    assert usage_data
    assert all(row["thread_id"] == "thread-new" for row in usage_data)
    assert query_data
    assert all(row["thread_id"] == "thread-new" for row in query_data)
