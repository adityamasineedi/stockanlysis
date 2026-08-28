"""Tests for prescan outcome log query helpers."""

from __future__ import annotations

import json
from pathlib import Path

from stockbot.portfolio_screener.outcome_log import (
    load_prescan_outcomes,
    query_prescan_outcomes,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_load_prescan_outcomes_latest_per_ticker(tmp_path: Path) -> None:
    path = tmp_path / "prescan_outcomes.jsonl"
    _write_rows(
        path,
        [
            {"ticker": "AAA", "quant_score": 50, "logged_at": "2026-01-01T00:00:00+00:00"},
            {"ticker": "AAA", "quant_score": 70, "logged_at": "2026-01-02T00:00:00+00:00"},
            {"ticker": "BBB", "quant_score": 80, "logged_at": "2026-01-01T00:00:00+00:00"},
        ],
    )
    rows = load_prescan_outcomes(path)
    assert len(rows) == 2
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["AAA"]["quant_score"] == 70


def test_query_prescan_outcomes_quality_and_analyze_ready(tmp_path: Path) -> None:
    path = tmp_path / "prescan_outcomes.jsonl"
    _write_rows(
        path,
        [
            {
                "ticker": "GOOD",
                "quality_score": 71,
                "growth_score": 40,
                "strength_score": 85,
                "quant_score": 63,
                "candidate_band": "WATCHLIST",
                "cash_conversion_status": "PASS",
                "hard_filter_status": "PASS",
                "verdict": "SECTOR_SPECIFIC_REVIEW",
                "logged_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "ticker": "WEAK",
                "quality_score": 40,
                "growth_score": 20,
                "strength_score": 50,
                "quant_score": 40,
                "candidate_band": "REMOVE",
                "cash_conversion_status": "ESCALATED_WATCH",
                "hard_filter_status": "PASS",
                "verdict": "HOLDING_MONITOR_ONLY",
                "logged_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "ticker": "EXCL",
                "quality_score": 80,
                "quant_score": 70,
                "candidate_band": "CANDIDATE",
                "cash_conversion_status": "PASS",
                "hard_filter_status": "HARD_EXCLUDE",
                "verdict": "NOT_SUITABLE_FOR_3Y_RESEARCH",
                "logged_at": "2026-01-01T00:00:00+00:00",
            },
        ],
    )
    rows = load_prescan_outcomes(path)
    matched = query_prescan_outcomes(rows, min_quality=65, analyze_ready_only=True)
    assert [r["ticker"] for r in matched] == ["GOOD"]
