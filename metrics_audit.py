"""Audit and reconciliation helpers for usage metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Any


METRICS_VERSION = "v1.0"
TOTAL_TOKENS_POLICY = "input_tokens + output_tokens"


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": "pass" if self.passed else "fail",
            "details": self.details,
        }


def _field(row: Any, name: str) -> Any:
    return getattr(row, name, None)


def _non_negative(value: Any) -> bool:
    return isinstance(value, int) and value >= 0


def _iter_token_rows(rows: list[Any]) -> list[Any]:
    return [row for row in rows if _field(row, "event_type") == "token_count"]


def _timestamp_to_utc_string(value: Any) -> str:
    if hasattr(value, "astimezone"):
        try:
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value or "")


def build_usage_audit_report(rows: list[Any], generated_at_utc: str) -> dict[str, Any]:
    token_rows = _iter_token_rows(rows)
    input_total = 0
    output_total = 0
    cached_total = 0
    reasoning_total = 0
    total_raw_sum = 0
    total_recomputed_sum = 0
    total_reported_sum = 0
    rows_with_raw_total = 0
    rows_with_recomputed_total = 0
    rows_with_both_totals = 0
    negative_rows = 0
    cached_over_input_rows = 0
    reasoning_over_output_rows = 0
    row_mismatch_count = 0
    mismatch_examples: list[dict[str, Any]] = []

    query_ids: set[str] = set()
    for row in token_rows:
        input_tokens = _field(row, "input_tokens")
        output_tokens = _field(row, "output_tokens")
        cached_input_tokens = _field(row, "cached_input_tokens")
        reasoning_output_tokens = _field(row, "reasoning_output_tokens")
        total_tokens_raw = _field(row, "total_tokens_raw")
        total_tokens_recomputed = _field(row, "total_tokens_recomputed")
        total_tokens = _field(row, "total_tokens")
        total_tokens_delta = _field(row, "total_tokens_delta")
        query_id = str(_field(row, "query_id") or "")

        if isinstance(input_tokens, int):
            input_total += input_tokens
        if isinstance(output_tokens, int):
            output_total += output_tokens
        if isinstance(cached_input_tokens, int):
            cached_total += cached_input_tokens
        if isinstance(reasoning_output_tokens, int):
            reasoning_total += reasoning_output_tokens
        if isinstance(total_tokens_raw, int):
            total_raw_sum += total_tokens_raw
            rows_with_raw_total += 1
        if isinstance(total_tokens_recomputed, int):
            total_recomputed_sum += total_tokens_recomputed
            rows_with_recomputed_total += 1
        if isinstance(total_tokens, int):
            total_reported_sum += total_tokens
        if isinstance(total_tokens_raw, int) and isinstance(total_tokens_recomputed, int):
            rows_with_both_totals += 1
        if total_tokens_delta not in (None, 0):
            row_mismatch_count += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "query_id": query_id or "unknown",
                        "thread_id": str(_field(row, "thread_id") or "unknown"),
                        "timestamp_utc": _timestamp_to_utc_string(_field(row, "timestamp")),
                        "delta": int(total_tokens_delta),
                        "raw_total": total_tokens_raw,
                        "recomputed_total": total_tokens_recomputed,
                    }
                )
        if query_id and query_id != "unknown":
            query_ids.add(query_id)

        token_values = [
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_output_tokens,
            total_tokens_raw,
            total_tokens_recomputed,
            total_tokens,
        ]
        if any(v is not None and not _non_negative(v) for v in token_values):
            negative_rows += 1

        if isinstance(cached_input_tokens, int) and isinstance(input_tokens, int) and cached_input_tokens > input_tokens:
            cached_over_input_rows += 1
        if (
            isinstance(reasoning_output_tokens, int)
            and isinstance(output_tokens, int)
            and reasoning_output_tokens > output_tokens
        ):
            reasoning_over_output_rows += 1

    total_delta = total_raw_sum - total_recomputed_sum
    query_count = len(query_ids)
    avg_tokens_per_query = total_reported_sum / query_count if query_count else 0.0
    cache_hit_rate = cached_total / input_total if input_total else 0.0
    output_input_ratio = output_total / input_total if input_total else 0.0

    checks = [
        CheckResult(
            check_id="A1_total_tokens_reconcile",
            passed=(rows_with_both_totals == 0 or total_delta == 0),
            details=(
                f"rows_with_both={rows_with_both_totals}, "
                f"sum_raw={total_raw_sum}, sum_recomputed={total_recomputed_sum}, delta={total_delta}"
            ),
        ),
        CheckResult(
            check_id="A2_cached_lte_input",
            passed=cached_over_input_rows == 0,
            details=f"rows_violating={cached_over_input_rows}",
        ),
        CheckResult(
            check_id="A3_reasoning_lte_output",
            passed=reasoning_over_output_rows == 0,
            details=f"rows_violating={reasoning_over_output_rows}",
        ),
        CheckResult(
            check_id="A4_non_negative_tokens",
            passed=negative_rows == 0,
            details=f"rows_violating={negative_rows}",
        ),
        CheckResult(
            check_id="A5_avg_tokens_per_query_definition",
            passed=(query_count == 0 and avg_tokens_per_query == 0.0)
            or abs((avg_tokens_per_query * query_count) - total_reported_sum) < 1e-9,
            details=f"query_count={query_count}, avg={avg_tokens_per_query:.6f}, total={total_reported_sum}",
        ),
    ]
    audit_status = "pass" if all(check.passed for check in checks) else "fail"
    failed_check_ids = [check.check_id for check in checks if not check.passed]

    action_items: list[dict[str, str]] = []
    if "A1_total_tokens_reconcile" in failed_check_ids:
        action_items.append(
            {
                "check_id": "A1_total_tokens_reconcile",
                "title": "Investigate total-token mismatch rows",
                "action": (
                    "Review rows where total_tokens_delta != 0 and confirm whether hidden token classes are present; "
                    "if total policy is input+output, fix upstream total_tokens emission."
                ),
            }
        )
    if "A2_cached_lte_input" in failed_check_ids:
        action_items.append(
            {
                "check_id": "A2_cached_lte_input",
                "title": "Fix cached-input overflow",
                "action": (
                    "Identify rows with cached_input_tokens > input_tokens and correct counter semantics or unit conversion "
                    "in upstream usage telemetry."
                ),
            }
        )
    if "A3_reasoning_lte_output" in failed_check_ids:
        action_items.append(
            {
                "check_id": "A3_reasoning_lte_output",
                "title": "Fix reasoning/output overlap definition",
                "action": (
                    "Ensure reasoning_output_tokens is either a documented subset of output_tokens or a separate metric "
                    "with non-overlapping definition."
                ),
            }
        )
    if "A4_non_negative_tokens" in failed_check_ids:
        action_items.append(
            {
                "check_id": "A4_non_negative_tokens",
                "title": "Correct negative token values",
                "action": (
                    "Backfill or drop corrupted rows with negative token values and add ingestion validation to reject them."
                ),
            }
        )
    if "A5_avg_tokens_per_query_definition" in failed_check_ids:
        action_items.append(
            {
                "check_id": "A5_avg_tokens_per_query_definition",
                "title": "Align avg/query denominator",
                "action": (
                    "Verify query_id attribution for token rows and ensure dashboard uses the same distinct query definition "
                    "as the audit report."
                ),
            }
        )

    return {
        "status": audit_status,
        "generated_at_utc": generated_at_utc,
        "timezone": "UTC",
        "metrics_version": METRICS_VERSION,
        "total_tokens_policy": TOTAL_TOKENS_POLICY,
        "notes": {
            "reasoning_output_tokens_semantics": (
                "May overlap with output_tokens depending on upstream telemetry definition."
            ),
        },
        "summary": {
            "token_row_count": len(token_rows),
            "rows_with_raw_total": rows_with_raw_total,
            "rows_with_recomputed_total": rows_with_recomputed_total,
            "rows_with_both_totals": rows_with_both_totals,
            "row_mismatch_count": row_mismatch_count,
            "input_tokens": input_total,
            "output_tokens": output_total,
            "cached_input_tokens": cached_total,
            "reasoning_output_tokens": reasoning_total,
            "total_tokens_reported": total_reported_sum,
            "total_tokens_raw_sum": total_raw_sum,
            "total_tokens_recomputed_sum": total_recomputed_sum,
            "total_tokens_delta_raw_minus_recomputed": total_delta,
            "cache_hit_rate": cache_hit_rate,
            "output_input_ratio": output_input_ratio,
            "distinct_query_count": query_count,
            "avg_tokens_per_query": avg_tokens_per_query,
            "mismatch_examples": mismatch_examples,
        },
        "checks": [check.to_dict() for check in checks],
        "failed_check_ids": failed_check_ids,
        "action_items": action_items,
    }
