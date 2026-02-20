from __future__ import annotations

from pathlib import Path

import pytest

from source_format_drift_check import SourceFormatDriftError, assert_no_source_format_drift

FIXTURE_BASE_DIR = Path(__file__).resolve().parent / "fixtures" / "source_format"
OK_FIXTURE_DIR = FIXTURE_BASE_DIR / "ok"
DRIFT_FIXTURE_DIR = FIXTURE_BASE_DIR / "drift"


def test_source_format_drift_check_accepts_expected_contract() -> None:
    assert_no_source_format_drift(OK_FIXTURE_DIR)


def test_source_format_drift_check_reports_clear_summary() -> None:
    with pytest.raises(SourceFormatDriftError) as exc_info:
        assert_no_source_format_drift(DRIFT_FIXTURE_DIR)

    summary = str(exc_info.value)
    assert "Source format drift detected" in summary
    assert "drift.jsonl" in summary
    assert "unknown_entry_type" in summary
    assert "invalid_timestamp" in summary
    assert "token_usage_path_missing" in summary
    assert "function_call_missing_call_id" in summary
    assert "unknown_response_item_type" in summary
    assert "function_call_output_missing_output" in summary
