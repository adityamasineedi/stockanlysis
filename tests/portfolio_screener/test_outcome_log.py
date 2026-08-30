"""Tests for prescan outcome log query helpers."""

from __future__ import annotations

import json
from pathlib import Path

from stockbot.portfolio_screener.outcome_log import (
    CandidatesFilter,
    build_candidates_messages,
    format_prescan_telegram_chunks,
    load_prescan_outcomes,
    parse_candidates_filter,
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


def test_query_prescan_outcomes_sorted_by_overall_not_quality() -> None:
    """The headline shows Overall first and bold, so the list must be ordered
    by Overall — sorting by Quality instead put a higher-Overall name below a
    lower-Overall one, which read as an unsorted/broken list."""
    rows = [
        {"ticker": "HEROMOTOCO", "quality_score": 77.9, "quant_score": 82.0},
        {"ticker": "BEL", "quality_score": 90.9, "quant_score": 56.0},
        {"ticker": "CYIENT", "quality_score": 42.4, "quant_score": 58.0},
    ]
    matched = query_prescan_outcomes(rows)
    assert [r["ticker"] for r in matched] == ["HEROMOTOCO", "CYIENT", "BEL"]


def test_prescan_row_html_does_not_round_across_tier_boundary() -> None:
    """VGUARD (60.3, Watchlist) and ADVENZYMES (59.6, Below threshold) must
    not both display "60" — that erases the actual tier boundary between
    them. See _format_prescan_row_html's quant_txt formatting."""
    from stockbot.portfolio_screener.outcome_log import _format_prescan_row_html

    vguard = _format_prescan_row_html(
        {"ticker": "VGUARD", "quant_score": 60.3, "candidate_band": "WATCHLIST"}
    )
    advenzymes = _format_prescan_row_html(
        {"ticker": "ADVENZYMES", "quant_score": 59.6, "candidate_band": "REMOVE"}
    )
    assert "60.3" in vguard
    assert "59.6" in advenzymes
    # The bug was both collapsing to a bare "60" under different tier headings.
    assert "60" not in vguard.replace("60.3", "")
    assert "60" not in advenzymes.replace("59.6", "")


def test_chunks_group_by_tier_without_repeating_the_label() -> None:
    """The tier used to be reprinted on all 18 rows. It is a section heading
    now, so each label appears once and sections run best-tier first."""
    rows = [
        {"ticker": "HEROMOTOCO", "quant_score": 82.3, "candidate_band": "STRONG_CANDIDATE"},
        {"ticker": "DBCORP", "quant_score": 63.1, "candidate_band": "WATCHLIST"},
        {"ticker": "VGUARD", "quant_score": 60.3, "candidate_band": "WATCHLIST"},
        {"ticker": "BEL", "quant_score": 56.4, "candidate_band": "REMOVE"},
    ]
    body = format_prescan_telegram_chunks(rows, title="All")[0]
    assert body.count("WATCHLIST (60–69)") == 1
    assert (
        body.index("TOP TIER (80+)")
        < body.index("WATCHLIST (60–69)")
        < body.index("BELOW THRESHOLD")
    )
    # Grouping must not reshuffle rows — they keep the Overall-descending
    # order query_prescan_outcomes already put them in.
    assert body.index("DBCORP") < body.index("VGUARD")


def test_chunks_split_without_orphaning_a_tier_heading() -> None:
    """When a tier spills into a second message, the heading and the <pre>
    wrapper must be reopened — a bare row block would lose its context."""
    rows = [
        {"ticker": f"TICKER{i:02d}", "quant_score": 70.0 - i, "candidate_band": "CANDIDATE"}
        for i in range(12)
    ]
    chunks = format_prescan_telegram_chunks(rows, title="All", max_len=500)
    assert len(chunks) > 1
    for chunk in chunks:
        if "<pre>" in chunk:
            assert chunk.count("<pre>") == chunk.count("</pre>")
            assert "GOOD PICK (70–79)" in chunk


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


def test_parse_candidates_filter_strong_band() -> None:
    parsed = parse_candidates_filter(["strong"])
    assert isinstance(parsed, CandidatesFilter)
    assert parsed.bands == {"STRONG_CANDIDATE"}
    assert parsed.analyze_ready_only is True


def test_parse_candidates_filter_quality() -> None:
    parsed = parse_candidates_filter(["quality", "65"])
    assert isinstance(parsed, CandidatesFilter)
    assert parsed.min_quality == 65.0


def test_format_prescan_telegram_chunks_includes_ticker() -> None:
    rows = [
        {
            "ticker": "HEROMOTOCO",
            "quality_score": 72.0,
            "growth_score": 55.0,
            "strength_score": 88.0,
            "quant_score": 82.24,
            "candidate_band": "STRONG_CANDIDATE",
            "cash_conversion_status": "PASS",
            "verdict": "AUTO_DEEP_ANALYSIS",
        }
    ]
    chunks = format_prescan_telegram_chunks(rows, title="Strong")
    assert len(chunks) == 1
    body = chunks[0]
    # One monospace row carries ticker, Overall and the three pillars.
    assert "HEROMOTOCO  82.2  Q72 G55 S88" in body
    # The tier is a section heading now, not repeated on the row itself.
    assert "<b>TOP TIER (80+)</b>" in body
    assert body.count("TOP TIER") == 1
    # Route shows as a trailing icon explained once in the footer legend.
    assert "✅" in body
    assert "✅ ready for /analyze" in body
    # A healthy cash gate is the common case and stays out of the list.
    assert "💚" not in body
    assert "Cash flow OK" not in body


def test_format_prescan_telegram_chunks_missing_qgs(monkeypatch) -> None:
    rows = [
        {
            "ticker": "HEROMOTOCO",
            "quant_score": 82.0,
            "candidate_band": "STRONG_CANDIDATE",
            "cash_conversion_status": "PASS",
            "verdict": "AUTO_DEEP_ANALYSIS",
        }
    ]
    monkeypatch.setattr(
        "stockbot.portfolio_screener.outcome_log.backfill_rows_qgs",
        lambda matched, persist=True: matched,
    )
    chunks = format_prescan_telegram_chunks(rows, title="Strong")
    # Missing pillars render as dashes rather than being silently omitted,
    # so the row still lines up with the rows that do have scores.
    assert "Q – G – S –" in chunks[0]
    assert "82.0" in chunks[0]


def test_backfill_row_qgs_persists_scores(monkeypatch, tmp_path: Path) -> None:
    from stockbot.portfolio_screener import outcome_log

    monkeypatch.setattr(outcome_log, "OUTCOMES_PATH", tmp_path / "prescan_outcomes.jsonl")

    import stockbot.fetch.tickers as tickers_mod
    import stockbot.portfolio_screener.data_loader as data_loader_mod
    import stockbot.portfolio_screener.quant_engine as quant_engine_mod

    class FakeComponents:
        business_quality = 77.9
        growth = 68.0
        financial_strength = 84.1

    class FakeQuant:
        components = FakeComponents()

    monkeypatch.setattr(
        tickers_mod,
        "resolve_ticker",
        lambda query, table: type("T", (), {"symbol": "HEROMOTOCO"})(),
    )
    monkeypatch.setattr(tickers_mod, "load_symbol_table", lambda: None)
    monkeypatch.setattr(data_loader_mod, "fetch_universe_metrics", lambda tickers: [object()])
    monkeypatch.setattr(quant_engine_mod, "compute_quant_score", lambda m, cfg: FakeQuant())

    row = {
        "ticker": "HEROMOTOCO",
        "quant_score": 82.0,
        "verdict": "AUTO_DEEP_ANALYSIS",
        "hard_filter_status": "PASS",
        "candidate_band": "STRONG_CANDIDATE",
        "cash_conversion_status": "PASS",
    }
    enriched = outcome_log.backfill_row_qgs(row, persist=True)
    assert enriched["quality_score"] == 77.9
    assert enriched["growth_score"] == 68.0
    assert enriched["strength_score"] == 84.1
    assert (tmp_path / "prescan_outcomes.jsonl").exists()


def test_build_candidates_messages_empty_log(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    chunks, err = build_candidates_messages([], path=missing)
    assert chunks == []
    assert err is not None
    assert "No prescan log" in err
