"""data_readiness.py — preflight gates and fallback chains."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from stockbot.data_readiness import (
    FALLBACK_CHAINS,
    apply_data_fallbacks,
    assess_data_readiness,
)
from stockbot.models import (
    ArBusinessSummary,
    Brief,
    Financials,
    NewsItems,
    PriceData,
    ReportText,
    Shareholding,
    Technicals,
    TickerInfo,
)

NOW = datetime.now(UTC)
TODAY = date(2026, 8, 31)
TICKER = TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co Limited", isin=None)
DEFAULT_SHAREHOLDING = Shareholding(50.0, None, 10.0, 5.0, "Q1", "NSE", NOW)
DEFAULT_NEWS = NewsItems([], [], [], [], "google_rss", NOW)
DEFAULT_AR_SECTIONS = {"Independent Auditor's Report": "clean opinion"}
_SENTINEL = object()


def _financials(*, years: int = 6, description: str | None = "Makes widgets.") -> Financials:
    cols = [f"FY{20 + index}" for index in range(years)]
    pnl = pd.DataFrame({col: [100.0 + index] for col, index in zip(cols, range(years), strict=True)}, index=["Net Profit"])
    empty = pd.DataFrame()
    return Financials(
        pnl=pnl,
        balance_sheet=empty,
        cash_flow=empty,
        ratios=empty,
        quarterly=empty,
        basis="consolidated",
        years_available=years,
        source="screener:consolidated",
        fetched_at=NOW,
        business_description=description,
    )


def _brief(
    *,
    financials: Financials | None | object = _SENTINEL,
    shareholding: Shareholding | None | object = _SENTINEL,
    news: NewsItems | None | object = _SENTINEL,
    ar_sections: dict[str, str] | None | object = _SENTINEL,
    missing: list[str] | None = None,
) -> Brief:
    if financials is _SENTINEL:
        financials = _financials()
    if shareholding is _SENTINEL:
        shareholding = DEFAULT_SHAREHOLDING
    if news is _SENTINEL:
        news = DEFAULT_NEWS
    if ar_sections is _SENTINEL:
        ar_sections = DEFAULT_AR_SECTIONS
    df = pd.DataFrame({"Close": [100.0]})
    sections = ar_sections if ar_sections is not None else {}
    return Brief(
        ticker=TICKER,
        price=PriceData(100.0, TODAY, df, df, 120.0, 80.0, "yfinance:TEST.NS", NOW),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], TODAY, "computed", NOW),
        financials=financials,
        shareholding=shareholding,
        news=news,
        annual_report=ReportText(
            sections=sections,
            report_year=2026,
            source_url="https://example/ar.pdf",
            truncated=False,
            dropped_sections=[],
            source="nse_annual_reports",
            fetched_at=NOW,
            business_summary=ArBusinessSummary(order_book_cr=100.0),
        ),
        missing=missing or [],
        token_count=1000,
        confidence_ceiling=10,
        generated_at=NOW,
    )


def test_assess_ready_when_core_fields_present():
    report = assess_data_readiness(_brief())
    assert report.ready_for_llm is True
    assert not report.blockers


def test_assess_blocks_without_financials():
    report = assess_data_readiness(_brief(financials=None))
    assert report.ready_for_llm is False
    assert any("Financial statements missing" in b for b in report.blockers)


def test_assess_blocks_without_annual_report():
    report = assess_data_readiness(_brief(ar_sections={}))
    assert report.ready_for_llm is False
    assert any("Annual report" in b for b in report.blockers)


def test_assess_blocks_thin_history_without_business_context():
    report = assess_data_readiness(
        _brief(financials=_financials(years=4, description=None), ar_sections={})
    )
    assert report.ready_for_llm is False


def test_fallback_chains_documented_for_all_core_fields():
    for name in ("price", "financials", "business_description", "shareholding", "annual_report", "news"):
        assert name in FALLBACK_CHAINS
        assert len(FALLBACK_CHAINS[name]) >= 1


def test_apply_fallbacks_uses_ar_excerpt_for_missing_description(monkeypatch):
    brief = _brief(
        financials=_financials(description=None),
        ar_sections={"Management Discussion": "Exports grew 18% in FY26 across enzyme segments."},
    )
    monkeypatch.setattr(
        "stockbot.data_readiness.fetch_shareholding",
        lambda *args, **kwargs: brief.shareholding,
    )
    monkeypatch.setattr(
        "stockbot.data_readiness.fetch_news",
        lambda *args, **kwargs: brief.news,
    )
    updated, attempts = apply_data_fallbacks(brief, TICKER)
    assert updated.financials is not None
    assert updated.financials.business_description is not None
    assert "[AR excerpt]" in updated.financials.business_description
    assert any(a.ok and a.source == "annual report MD&A excerpt" for a in attempts)
