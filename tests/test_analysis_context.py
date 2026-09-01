"""Tests for peer / sector scorecard / portfolio execution enrichment."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from stockbot.analysis.analysis_context import (
    build_peer_snapshot,
    build_portfolio_execution,
    build_sector_scorecard,
    execution_pm_for_verdict,
)
from stockbot.models import (
    Brief,
    BriefMetadata,
    PortfolioExecutionContext,
    PrescanSummary,
    PriceData,
    ReportText,
    SectorScorecardContext,
    Technicals,
    TickerInfo,
)

NOW = datetime.now(UTC)
TODAY = datetime.now(UTC).date()


def _minimal_brief(**kwargs) -> Brief:
    ticker = TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co", isin=None)
    ohlcv = pd.DataFrame(
        {"Close": [100.0], "Low": [99.0], "High": [101.0]},
        index=pd.date_range("2026-01-01", periods=1),
    )
    price = PriceData(
        current_price_abs=100.0,
        price_date=TODAY,
        ohlcv_adjusted=ohlcv,
        ohlcv_unadjusted=ohlcv,
        week52_low_abs=80.0,
        week52_high_abs=120.0,
        source="test",
        fetched_at=NOW,
    )
    technicals = Technicals(
        100.0,
        95.0,
        50.0,
        [90.0],
        [110.0],
        TODAY,
        "computed",
        NOW,
        trend_label="uptrend",
        price_vs_bollinger="inside_bands",
    )
    annual_report = ReportText(
        sections={},
        report_year=2025,
        source_url=None,
        truncated=False,
        dropped_sections=[],
        source="test",
        fetched_at=NOW,
        business_summary=None,
    )
    base = Brief(
        ticker=ticker,
        price=price,
        technicals=technicals,
        financials=None,
        shareholding=None,
        news=None,
        annual_report=annual_report,
        missing=[],
        token_count=100,
        confidence_ceiling=10,
        generated_at=NOW,
    )
    return kwargs.get("brief", base) if "brief" in kwargs else base


def test_build_peer_snapshot_percentile(monkeypatch: pytest.MonkeyPatch):
    metadata = BriefMetadata(
        ticker="TEST",
        company_name="Test",
        sector="Technology",
        industry="Software",
        market_cap_cr=1000.0,
        ttm_pe=20.0,
        ttm_pb=3.0,
        price=100.0,
        price_date=TODAY.isoformat(),
        range_52w_low=80.0,
        range_52w_high=120.0,
        rsi_14=50.0,
        pe_price_eps=20.0,
    )

    def fake_meta(symbol: str) -> dict:
        if symbol == "TEST":
            return {"sector": "Technology", "trailing_pe": 20.0, "roe_pct": 18.0}
        if symbol == "PEER1":
            return {"sector": "Technology", "trailing_pe": 30.0}
        if symbol == "PEER2":
            return {"sector": "Technology", "trailing_pe": 10.0}
        return {"sector": "Energy", "trailing_pe": 12.0}

    monkeypatch.setattr(
        "stockbot.analysis.analysis_context._load_universe_symbols",
        lambda: {"PEER1", "PEER2", "OTHER"},
    )
    monkeypatch.setattr(
        "stockbot.analysis.analysis_context.fetch_market_metadata",
        fake_meta,
    )

    snap = build_peer_snapshot("TEST", metadata)
    assert snap is not None
    assert snap.peer_count == 2
    assert snap.pe_percentile == pytest.approx(100.0)
    assert snap.target_roe_pct == pytest.approx(18.0)


def test_build_sector_scorecard_for_insurer():
    prescan = PrescanSummary(
        quant_score=18.3,
        quality_score=10.0,
        growth_score=20.0,
        strength_score=15.0,
        band="PASS",
        issuer_class="INSURER",
        route="SECTOR_SPECIFIC_REVIEW",
        eligibility_verdict="SECTOR_SPECIFIC_REVIEW",
        cash_conversion_status="NOT_APPLICABLE",
        ocf_pat_current=None,
        ocf_pat_3y=None,
        data_confidence="HIGH",
        major_flags=(),
    )
    annual_report = ReportText(
        sections={
            "Management Discussion": "Our combined ratio improved to 98% while solvency remains strong."
        },
        report_year=2025,
        source_url=None,
        truncated=False,
        dropped_sections=[],
        source="test",
        fetched_at=NOW,
        business_summary=None,
    )
    brief = _minimal_brief()
    brief = Brief(
        ticker=brief.ticker,
        price=brief.price,
        technicals=brief.technicals,
        financials=brief.financials,
        shareholding=brief.shareholding,
        news=brief.news,
        annual_report=annual_report,
        missing=brief.missing,
        token_count=brief.token_count,
        confidence_ceiling=brief.confidence_ceiling,
        generated_at=brief.generated_at,
    )
    card = build_sector_scorecard(brief, prescan)
    assert card is not None
    assert card.issuer_class == "INSURER"
    assert "combined ratio" in card.scorecard_lens.lower() or "Insurer" in card.scorecard_lens
    assert card.generic_quant_note is not None
    assert "18.3" in card.generic_quant_note


def test_build_portfolio_execution_defaults():
    ctx = build_portfolio_execution("UNKNOWN", None)
    assert ctx.max_position_pct == 10.0
    assert "delivery" in ctx.delivery_note.lower()
    assert ctx.in_sip_portfolio is False


def test_execution_pm_for_verdict_merges_fields():
    pm = execution_pm_for_verdict(
        None,
        SectorScorecardContext("BANK", "bank lens", (), ()),
        PortfolioExecutionContext(
            True,
            "Core",
            10000.0,
            2500.0,
            10.0,
            1,
            "quarterly",
            "delivery only",
            "concentration warning",
        ),
        "uptrend",
        "inside_bands",
    )
    assert pm["sector_scorecard_issuer"] == "BANK"
    assert pm["suggested_tranche_inr"] == 2500.0
    assert pm["trend_label"] == "uptrend"
