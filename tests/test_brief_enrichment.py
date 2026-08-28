"""Tests for brief enrichment blocks."""

from __future__ import annotations

from datetime import UTC, date, datetime

from stockbot.brief_enrichment import (
    build_news_summary,
    build_street_consensus,
    format_metadata_json,
    stage2_mode_from_prescan,
)
from stockbot.models import BriefMetadata, NewsItems, NewsSummaryItem, PrescanSummary, RedFlag, TickerInfo


def test_build_news_summary_ranks_order_book_headline():
    news = NewsItems(
        general=[
            RedFlag(
                headline="Company wins ₹500cr defence contract",
                url="https://example.com/a",
                published_date=date(2026, 8, 20),
                found_by_query="BEL",
            ),
            RedFlag(
                headline="Routine AGM notice - Economic Times",
                url="https://example.com/b",
                published_date=date(2026, 8, 21),
                found_by_query="BEL",
            ),
        ],
        red_flags=[],
        queries_run=[],
        queries_empty=[],
        source="google_news_rss",
        fetched_at=datetime.now(UTC),
    )
    summary = build_news_summary(news)
    assert len(summary) >= 1
    assert summary[0].news_type == "order_book"


def test_stage2_mode_from_prescan_defence_is_full():
    summary = PrescanSummary(
        quant_score=57.0,
        quality_score=91.0,
        growth_score=80.0,
        strength_score=84.0,
        band="REMOVE",
        issuer_class="DEFENCE_EPC_PROJECT",
        route="DEFENCE_WC_REVIEW",
        eligibility_verdict="SECTOR_SPECIFIC_REVIEW",
        cash_conversion_status="ESCALATED_WATCH",
        ocf_pat_current=0.25,
        ocf_pat_3y=0.39,
        data_confidence="HIGH",
        major_flags=("cfo_pat_multi_year_weak",),
    )
    mode, reasons = stage2_mode_from_prescan(summary)
    assert mode == "FULL"
    assert any("issuer_class" in r for r in reasons)


def test_stage2_mode_from_prescan_clean_auto_deep_is_lite():
    summary = PrescanSummary(
        quant_score=72.0,
        quality_score=75.0,
        growth_score=70.0,
        strength_score=78.0,
        band="CANDIDATE",
        issuer_class="NON_FINANCIAL",
        route="AUTO_DEEP",
        eligibility_verdict="AUTO_DEEP_ANALYSIS",
        cash_conversion_status="PASS",
        ocf_pat_current=0.9,
        ocf_pat_3y=0.85,
        data_confidence="HIGH",
        major_flags=(),
    )
    mode, _ = stage2_mode_from_prescan(summary)
    assert mode == "LITE"


def test_format_metadata_json_includes_sector():
    from stockbot.models import BriefMetadata

    text = format_metadata_json(
        BriefMetadata(
            ticker="BEL",
            company_name="BEL",
            sector="Industrials",
            industry="Aerospace & Defense",
            market_cap_cr=300000.0,
            ttm_pe=49.0,
            ttm_pb=12.0,
            price=408.55,
            price_date="2026-08-28",
            range_52w_low=361.2,
            range_52w_high=473.45,
            rsi_14=54.7,
        )
    )
    assert '"sector": "Industrials"' in text
    assert '"ttm_pe": 49.0' in text
    assert '"street_consensus"' in text


def test_build_street_consensus_from_yfinance_metadata(monkeypatch):
    from datetime import UTC, datetime

    from stockbot.models import PriceData

    import pandas as pd

    monkeypatch.setattr(
        "stockbot.brief_enrichment.fetch_market_metadata",
        lambda symbol: {
            "analyst_count": 12,
            "recommendation_key": "buy",
            "target_mean_price": 500.0,
            "target_low_price": 420.0,
            "target_high_price": 560.0,
        },
    )
    price = PriceData(
        550.0,
        date(2026, 8, 28),
        pd.DataFrame({"Close": [550.0]}),
        pd.DataFrame({"Close": [550.0]}),
        600.0,
        400.0,
        "yfinance",
        datetime.now(UTC),
    )
    consensus = build_street_consensus(
        TickerInfo("BEL", "NSE", "BEL", None),
        price,
    )
    assert consensus.target_mean_price == 500.0
    assert consensus.price_vs_target_pct == 10.0
    assert consensus.tension == "MEDIUM"
