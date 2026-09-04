"""Tests for unified product universe and portfolio progress."""

from __future__ import annotations

import json
from pathlib import Path

from stockbot.portfolio_progress import (
    build_portfolio_progress,
    format_daily_tips_html,
    format_portfolio_progress_html,
    select_daily_tips,
)
from stockbot.product_universe import load_product_universe
from stockbot.sector_map import clear_sector_map_cache, sector_for_symbol


def test_product_universe_unifies_watchlist_and_sip() -> None:
    uni = load_product_universe()
    tickers = set(uni.tickers)
    assert "RELIANCE" in tickers
    assert "MAZDOCK" in tickers
    assert "TCS" in tickers
    # After unify, SIP names live on the watchlist file too → "both" sources.
    mazdock = next(r for r in uni.symbols if r.symbol == "MAZDOCK")
    assert "sip" in mazdock.sources
    assert mazdock.sip_bucket_id is not None
    assert len(uni.tickers) >= 60


def test_sector_map_covers_universe() -> None:
    clear_sector_map_cache()
    assert sector_for_symbol("TCS") == "Technology"
    assert sector_for_symbol("MAZDOCK") == "Industrials"
    assert sector_for_symbol("UNKNOWNXYZ") == "Unclassified"


def test_select_daily_tips_prefers_analyze_now(tmp_path: Path, monkeypatch) -> None:
    from stockbot.portfolio_screener import outcome_log
    from stockbot.product_universe import ProductUniverse, UniverseSymbol

    path = tmp_path / "prescan_outcomes.jsonl"
    rows = [
        {
            "ticker": "LATER",
            "quant_score": 90,
            "quality_score": 80,
            "growth_score": 70,
            "strength_score": 70,
            "hard_filter_status": "PASS",
            "cash_conversion_status": "PASS",
            "verdict": "HOLDING_MONITOR_ONLY",
        },
        {
            "ticker": "NOW",
            "quant_score": 60,
            "quality_score": 70,
            "growth_score": 60,
            "strength_score": 60,
            "hard_filter_status": "PASS",
            "cash_conversion_status": "PASS",
            "verdict": "AUTO_DEEP_ANALYSIS",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(outcome_log, "OUTCOMES_PATH", path)

    uni = ProductUniverse(
        symbols=(
            UniverseSymbol("NOW", frozenset({"watchlist"})),
            UniverseSymbol("LATER", frozenset({"watchlist"})),
        ),
        watchlist_path=tmp_path / "wl.txt",
        sip_path=None,
    )
    tips = select_daily_tips(limit=2, universe=uni, pick_rows=rows)
    assert [r["ticker"] for r in tips] == ["NOW", "LATER"]
    html = format_daily_tips_html(tips)
    assert "NOW" in html
    assert "/analyze NOW" in html


def test_portfolio_progress_counts_stages(monkeypatch, tmp_path: Path) -> None:
    from stockbot import storage
    from stockbot.product_universe import ProductUniverse, UniverseSymbol

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "prog.db")
    storage.save_holding(7, "HELD", 10, 100.0)
    storage.save_analysis(
        ticker="BUYME",
        verdict_json={
            "verdict": "BUY ON CORRECTION",
            "buy_range_allowed": True,
            "buy_zone_abs": [90.0, 100.0],
            "current_price_abs": 110.0,
        },
        report_md="# report",
        brief_text="brief",
        stage1_tokens=1,
        stage2_tokens=1,
        cost_inr=1.0,
        validation_passed=True,
        missing=[],
    )

    uni = ProductUniverse(
        symbols=(
            UniverseSymbol("BUYME", frozenset({"watchlist", "sip"}), "b1", "Bucket"),
            UniverseSymbol("HELD", frozenset({"watchlist"})),
            UniverseSymbol("PLAIN", frozenset({"watchlist"})),
        ),
        watchlist_path=tmp_path / "wl.txt",
        sip_path=None,
    )
    monkeypatch.setattr(
        "stockbot.portfolio_progress.load_prescan_outcomes",
        list,
    )
    monkeypatch.setattr(
        "stockbot.portfolio_progress.query_pick_outcomes",
        lambda rows: [],
    )
    report = build_portfolio_progress(7, universe=uni)
    assert report.universe_size == 3
    assert report.held_count == 1
    assert report.buy_range_count == 1
    assert report.analyzed_count == 1
    html = format_portfolio_progress_html(report, universe=uni)
    assert "BUYME" in html
    assert "HELD" in html
    assert "12–18" in html


def test_portfolio_progress_includes_off_universe_soft_picks(monkeypatch, tmp_path: Path) -> None:
    """/prescan names not on watchlist must still show under soft picks."""
    from stockbot import storage
    from stockbot.product_universe import ProductUniverse, UniverseSymbol

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "prog2.db")
    uni = ProductUniverse(
        symbols=(UniverseSymbol("TCS", frozenset({"watchlist"})),),
        watchlist_path=tmp_path / "wl.txt",
        sip_path=None,
    )
    pick_row = {
        "ticker": "PRECWIRE",
        "quant_score": 54.0,
        "quality_score": 60.0,
        "growth_score": 50.0,
        "strength_score": 50.0,
        "hard_filter_status": "PASS",
        "cash_conversion_status": "PASS",
        "verdict": "HOLDING_MONITOR_ONLY",
    }
    monkeypatch.setattr(
        "stockbot.portfolio_progress.load_prescan_outcomes",
        lambda: [pick_row],
    )
    monkeypatch.setattr(
        "stockbot.portfolio_progress.query_pick_outcomes",
        lambda rows: [pick_row],
    )
    report = build_portfolio_progress(None, universe=uni)
    assert report.soft_pick_count == 1
    assert report.off_universe_count == 1
    prec = next(r for r in report.rows if r.symbol == "PRECWIRE")
    assert prec.soft_pick is True
    assert prec.in_universe is False
    html = format_portfolio_progress_html(report, universe=uni)
    assert "PRECWIRE" in html
    assert "off-list" in html
    assert "soft picks" in html.lower() or "Soft pick" in html
