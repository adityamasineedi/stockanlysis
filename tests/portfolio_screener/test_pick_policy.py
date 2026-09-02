"""Tests for soft pick policy."""

from __future__ import annotations

import json
from pathlib import Path

from stockbot.portfolio_screener.outcome_log import (
    build_pick_messages,
    parse_candidates_filter,
)
from stockbot.portfolio_screener.pick_policy import (
    is_pick_eligible,
    pick_skip_reason,
    pick_tier,
    query_pick_outcomes,
)


def _row(**kwargs: object) -> dict:
    base = {
        "ticker": "TEST",
        "quant_score": 55.0,
        "quality_score": 60.0,
        "growth_score": 45.0,
        "strength_score": 65.0,
        "hard_filter_status": "PASS",
        "cash_conversion_status": "PASS",
        "verdict": "SECTOR_SPECIFIC_REVIEW",
    }
    base.update(kwargs)
    return base


def test_pick_eligible_on_quant_floor() -> None:
    row = _row(quant_score=50.0)
    assert is_pick_eligible(row)
    assert pick_skip_reason(row) is None


def test_pick_eligible_on_strength_pillar_despite_low_quant() -> None:
    row = _row(
        ticker="PRAJIND",
        quant_score=33.0,
        strength_score=79.0,
        verdict="HOLDING_MONITOR_ONLY",
    )
    assert is_pick_eligible(row)
    assert pick_tier(row) == "analyze_if_interested"


def test_pick_rejects_critical_cash() -> None:
    row = _row(cash_conversion_status="CRITICAL", quant_score=80.0)
    assert not is_pick_eligible(row)
    assert pick_skip_reason(row) == "cash conversion CRITICAL"


def test_pick_rejects_hard_exclude() -> None:
    row = _row(
        hard_filter_status="HARD_EXCLUDE",
        verdict="NOT_SUITABLE_FOR_3Y_RESEARCH",
        quant_score=80.0,
    )
    assert not is_pick_eligible(row)


def test_pick_rejects_weak_without_pillar() -> None:
    row = _row(
        quant_score=40.0,
        quality_score=40.0,
        growth_score=20.0,
        strength_score=50.0,
        verdict="HOLDING_MONITOR_ONLY",
        cash_conversion_status="ESCALATED_WATCH",
    )
    assert not is_pick_eligible(row)
    reason = pick_skip_reason(row)
    assert reason is not None
    assert "40.0" in reason


def test_pick_quality_override_passes() -> None:
    row = _row(quant_score=35.0, quality_override=True, verdict="HOLDING_MONITOR_ONLY")
    assert is_pick_eligible(row)


def test_query_pick_outcomes_orders_analyze_now_first() -> None:
    rows = [
        _row(ticker="MONITOR", quant_score=72.0, verdict="HOLDING_MONITOR_ONLY"),
        _row(ticker="AUTO", quant_score=68.0, verdict="AUTO_DEEP_ANALYSIS"),
        _row(ticker="SECTOR", quant_score=55.0, verdict="SECTOR_SPECIFIC_REVIEW"),
    ]
    matched = query_pick_outcomes(rows)
    assert [r["ticker"] for r in matched] == ["AUTO", "SECTOR", "MONITOR"]


def test_parse_candidates_filter_pick_mode() -> None:
    parsed = parse_candidates_filter(["pick"])
    assert parsed.pick_mode is True
    assert parsed.analyze_ready_only is False


def test_build_pick_messages_from_log(tmp_path: Path) -> None:
    path = tmp_path / "prescan_outcomes.jsonl"
    rows = [
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    chunks, err = build_pick_messages([], path=path)
    assert err is None
    assert len(chunks) == 1
    body = chunks[0]
    assert "GOOD" in body
    assert "WEAK" not in body
    assert "RUN /ANALYZE" in body


def test_build_pick_messages_daily_limits_to_two(tmp_path: Path) -> None:
    path = tmp_path / "prescan_outcomes.jsonl"
    rows = [
        {
            "ticker": "AUTO1",
            "quality_score": 71,
            "growth_score": 40,
            "strength_score": 85,
            "quant_score": 80,
            "candidate_band": "CANDIDATE",
            "cash_conversion_status": "PASS",
            "hard_filter_status": "PASS",
            "verdict": "AUTO_DEEP_ANALYSIS",
            "logged_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "ticker": "AUTO2",
            "quality_score": 70,
            "growth_score": 40,
            "strength_score": 80,
            "quant_score": 75,
            "candidate_band": "CANDIDATE",
            "cash_conversion_status": "PASS",
            "hard_filter_status": "PASS",
            "verdict": "AUTO_DEEP_ANALYSIS",
            "logged_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "ticker": "AUTO3",
            "quality_score": 69,
            "growth_score": 40,
            "strength_score": 78,
            "quant_score": 70,
            "candidate_band": "WATCHLIST",
            "cash_conversion_status": "PASS",
            "hard_filter_status": "PASS",
            "verdict": "AUTO_DEEP_ANALYSIS",
            "logged_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    chunks, err = build_pick_messages(["daily"], path=path)
    assert err is None
    body = chunks[0]
    assert "Today" in body or "tips" in body.lower()
    assert "AUTO1" in body
    assert "AUTO2" in body
    assert "AUTO3" not in body
    assert "/analyze AUTO1" in body
