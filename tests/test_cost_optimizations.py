"""Tests for Stage 1 input trimming and validation auto-fix."""

from datetime import UTC, date, datetime

import pandas as pd

from stockbot.llm.extract import (
    STAGE1_SECTION_HEADINGS,
    _select_stage1_sections,
    build_user_message,
)
from stockbot.models import (
    Brief,
    NewsItems,
    PriceData,
    RedFlag,
    ReportText,
    Technicals,
    TickerInfo,
)
from stockbot.validate import try_auto_fix_report, validate_report

NOW = datetime.now(UTC)
TICKER = TickerInfo(symbol="T", exchange="NSE", company_name="Test", isin=None)


def test_select_stage1_sections_filters_and_caps():
    sections = {
        "Independent Auditor's Report": "A" * 8000,
        "Key Audit Matters": "B" * 8000,
        "Related Party": "C" * 8000,
        "Noise Section": "should drop",
    }
    picked = _select_stage1_sections(sections)
    assert "Noise Section" not in picked
    assert set(picked) <= set(STAGE1_SECTION_HEADINGS)
    total_chars = sum(len(v) for v in picked.values())
    assert total_chars <= 15_000 * 4 + 500


def test_build_user_message_omits_general_news():
    news = NewsItems(
        general=[
            RedFlag(
                headline="routine AGM",
                url="https://example.com",
                published_date=date(2026, 1, 1),
                found_by_query="general",
            )
        ],
        red_flags=[
            RedFlag(
                headline="SEBI action",
                url="https://example.com/2",
                published_date=date(2026, 2, 1),
                found_by_query="sebi",
            )
        ],
        queries_run=["sebi"],
        queries_empty=[],
        source="google_news",
        fetched_at=NOW,
    )
    brief = Brief(
        ticker=TICKER,
        price=PriceData(100.0, date(2026, 8, 1), pd.DataFrame(), pd.DataFrame(), 110.0, 90.0, "yfinance", NOW),
        technicals=Technicals(None, None, None, [], [], date(2026, 8, 1), "computed", NOW),
        financials=None,
        shareholding=None,
        news=news,
        annual_report=ReportText({}, None, None, False, [], "nse", NOW),
        missing=[],
        token_count=100,
        confidence_ceiling=10,
        generated_at=NOW,
    )
    message = build_user_message(brief)
    assert "General (last 12 months)" not in message
    assert "SEBI action" in message
    assert "routine AGM" not in message


def test_auto_fix_confidence_scale():
    report = (
        "Quick verdict. Confidence: 5/7.\n\n"
        "**SHOULD I BUY?**\nNo.\n\n"
        "```json\n"
        '{"verdict":"WATCH","current_price_abs":100,"price_date":"2026-08-26",'
        '"buy_zone_abs":null,'
        '"valuation_inputs":{"eps_bear":10,"eps_base":12,"eps_bull":14,'
        '"multiple_bear":[10,12],"multiple_base":[14,16],"multiple_bull":[18,20]},'
        '"confidence":5,"risk":"LOW","business_quality":7,"financial_health":7,'
        '"management_quality":7,"earnings_quality":"HIGH","holding_period":"3y",'
        '"reasons_buy":["a"],"reasons_avoid":["b"],"biggest_watch":"c",'
        '"missing_data_impact":"none","gates_failed":[]}\n'
        "```\n\nResearch and education, not investment advice.\n"
    )
    brief = Brief(
        ticker=TICKER,
        price=PriceData(100.0, date(2026, 8, 26), pd.DataFrame(), pd.DataFrame(), 110.0, 90.0, "yfinance", NOW),
        technicals=Technicals(None, None, None, [], [], date(2026, 8, 26), "computed", NOW),
        financials=None,
        shareholding=None,
        news=None,
        annual_report=ReportText({}, None, None, False, [], "nse", NOW),
        missing=[],
        token_count=100,
        confidence_ceiling=10,
        generated_at=NOW,
    )
    failed = validate_report(report, brief, stage2_mode="LITE")
    assert not failed.passed
    fixed = try_auto_fix_report(report, failed, brief, stage2_mode="LITE")
    assert fixed is not None
    new_text, _new_result = fixed
    assert "5/10" in new_text
    assert "/7" not in new_text
