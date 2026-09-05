"""Tests for /analyze all one-shot batch planning."""

from __future__ import annotations

import json
from pathlib import Path

from stockbot.analyze_batch import (
    ANALYZE_ALL_USAGE,
    AnalyzeBatchRequest,
    format_batch_start_html,
    format_batch_summary_html,
    parse_analyze_all_args,
    plan_analyze_batch,
    row_tickers_for_tests,
)


def _row(
    ticker: str,
    quant: float,
    *,
    verdict: str = "AUTO_DEEP_ANALYSIS",
    cash: str = "PASS",
    band: str = "CANDIDATE",
) -> dict:
    return {
        "ticker": ticker,
        "quant_score": quant,
        "verdict": verdict,
        "cash_conversion_status": cash,
        "candidate_band": band,
        "hard_filter_status": "PASS",
        "logged_at": "2026-09-05T00:00:00+00:00",
    }


def test_parse_analyze_all_defaults_to_good_lite() -> None:
    req = parse_analyze_all_args([])
    assert isinstance(req, AnalyzeBatchRequest)
    assert req.scope == "good"
    assert req.min_quant == 70.0
    assert req.max_names == 12
    assert req.force_lite is True
    assert req.skip_already_analyzed is True


def test_parse_analyze_all_options() -> None:
    req = parse_analyze_all_args(["top", "full", "fresh", "force", "5"])
    assert isinstance(req, AnalyzeBatchRequest)
    assert req.scope == "top"
    assert req.min_quant == 80.0
    assert req.max_names == 5
    assert req.force_lite is False
    assert req.skip_cache is True
    assert req.force_gate is True
    assert req.skip_already_analyzed is False


def test_parse_analyze_all_help_and_unknown() -> None:
    assert parse_analyze_all_args(["help"]) == ANALYZE_ALL_USAGE
    err = parse_analyze_all_args(["nope"])
    assert isinstance(err, str)
    assert "Unknown option" in err


def test_row_tickers_respects_min_quant_and_ready() -> None:
    rows = [
        _row("AAA", 85.0, band="STRONG_CANDIDATE"),
        _row("BBB", 72.0),
        _row("CCC", 65.0, band="WATCHLIST"),
        _row("DDD", 90.0, verdict="NOT_SUITABLE"),
    ]
    good = row_tickers_for_tests(rows, AnalyzeBatchRequest())
    assert good == ["AAA", "BBB"]
    top = row_tickers_for_tests(
        rows, AnalyzeBatchRequest(scope="top", min_quant=80.0)
    )
    assert top == ["AAA"]
    ready = row_tickers_for_tests(
        rows, AnalyzeBatchRequest(scope="ready", min_quant=0.0, max_names=20)
    )
    assert ready == ["AAA", "BBB", "CCC"]


def test_plan_analyze_batch_skips_already_analyzed(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    rows = [
        _row("HEROMOTOCO", 83.5, band="STRONG_CANDIDATE"),
        _row("TCS", 73.2),
        _row("BEL", 62.8, band="WATCHLIST"),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    plan = plan_analyze_batch(AnalyzeBatchRequest(), path=path, already={"TCS"})
    assert not isinstance(plan, str)
    assert plan.tickers == ["HEROMOTOCO"]
    assert plan.skipped_analyzed == ["TCS"]

    start = format_batch_start_html(plan)
    assert "HEROMOTOCO" in start
    assert "Batch analyze" in start or "Batch" in start


def test_format_batch_summary_mentions_stop() -> None:
    html = format_batch_summary_html(
        planned=["AAA", "BBB"],
        completed=["AAA"],
        failed=[],
        skipped_gate=[],
        stopped=True,
        budget_stopped=False,
    )
    assert "AAA" in html
    assert "BBB" in html
    assert "/stop" in html or "Stopped" in html
