"""Module 8 (Stage 1 extraction) unit tests. Pure-logic pieces (prompt
construction, schema validation) are tested here without any network
call. Live-tested against a real brief and a real Sonnet 5 call — that
first live run surfaced a real bug (MAX_TOKENS too low, causing a silent
None instead of a parsed result on a large real input); see
test_run_stage1_raises_loudly_when_parsed_output_is_none below for the
regression test, and the MAX_TOKENS comment in extract.py for the story."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stockbot.llm.extract import (
    ExtractionResult,
    Stage1Error,
    build_user_message,
    run_stage1,
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

NOW = datetime.now(UTC)


def _minimal_brief(*, sections: dict[str, str], news: NewsItems | None) -> Brief:
    df = pd.DataFrame({"Close": [100.0]})
    return Brief(
        ticker=TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co Limited", isin=None),
        price=PriceData(100.0, date(2026, 8, 25), df, df, 120.0, 80.0, "yfinance", NOW),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], date(2026, 8, 25), "computed", NOW),
        financials=None,
        shareholding=None,
        news=news,
        annual_report=ReportText(
            sections=sections,
            report_year=2026,
            source_url="https://example.com/ar.pdf",
            truncated=False,
            dropped_sections=[],
            source="nse_annual_reports",
            fetched_at=NOW,
        ),
        missing=[],
        token_count=0,
        confidence_ceiling=10,
        generated_at=NOW,
    )


def test_extraction_result_defaults_are_empty_not_guessed():
    result = ExtractionResult()
    assert result.auditor_opinion_type is None
    assert result.auditor_concerns == []
    assert result.red_flags_found == []
    assert result.extraction_gaps == []


def test_extraction_result_validates_nested_red_flag_dataclass():
    result = ExtractionResult.model_validate(
        {
            "red_flags_found": [
                {
                    "headline": "h",
                    "url": "u",
                    "published_date": "2026-01-01",
                    "found_by_query": "q",
                }
            ]
        }
    )
    assert isinstance(result.red_flags_found[0], RedFlag)


def test_extraction_result_rejects_invalid_opinion_type():
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError, exact type not load-bearing here
        ExtractionResult.model_validate({"auditor_opinion_type": "not-a-real-value"})


def test_build_user_message_includes_annual_report_sections():
    brief = _minimal_brief(sections={"Key Audit Matters": "some audit text"}, news=None)
    message = build_user_message(brief)
    assert "Key Audit Matters" in message
    assert "some audit text" in message


def test_build_user_message_marks_missing_annual_report():
    brief = _minimal_brief(sections={}, news=None)
    message = build_user_message(brief)
    assert "MISSING: no annual report sections" in message


def test_build_user_message_marks_missing_news():
    brief = _minimal_brief(sections={"X": "y"}, news=None)
    message = build_user_message(brief)
    assert "MISSING: news fetch failed entirely" in message


def test_build_user_message_includes_queries_empty():
    news = NewsItems(
        general=[],
        red_flags=[],
        queries_run=["Q1", "Q2"],
        queries_empty=["Q2"],
        source="google_news_rss",
        fetched_at=NOW,
    )
    brief = _minimal_brief(sections={"X": "y"}, news=news)
    message = build_user_message(brief)
    assert "Queries with zero results" in message
    assert "Q2" in message


def test_build_user_message_caps_red_flags_per_query():
    items = [RedFlag(f"Headline {i}", f"url-{i}", date(2026, 1, 1), "Q1") for i in range(20)]
    news = NewsItems(
        general=[],
        red_flags=items,
        queries_run=["Q1"],
        queries_empty=[],
        source="google_news_rss",
        fetched_at=NOW,
    )
    brief = _minimal_brief(sections={"X": "y"}, news=news)
    message = build_user_message(brief)
    # 20 candidates, capped to 8 per query -> only 8 "Headline" lines
    assert message.count("Headline") == 8


def test_run_stage1_raises_loudly_when_parsed_output_is_none(monkeypatch, tmp_path):
    # Regression test for a real bug found on the first live run: a large
    # real brief caused the response to be cut off before any text block
    # was produced, so the SDK's parsed_output came back None with no
    # exception raised. run_stage1 must turn that into a loud, actionable
    # error rather than handing a None back to the caller.
    from stockbot.llm import fixtures as fixtures_module

    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)

    fake_response = MagicMock()
    fake_response.parsed_output = None
    fake_response.stop_reason = "max_tokens"
    fake_response.usage.input_tokens = 1000
    fake_response.usage.output_tokens = 4096
    fake_response.usage.cache_read_input_tokens = 0

    fake_client = MagicMock()
    fake_client.messages.parse.return_value = fake_response

    monkeypatch.setattr("stockbot.llm.client.log_call", lambda **kwargs: 12.5)

    brief = _minimal_brief(sections={"X": "y"}, news=None)
    with pytest.raises(Stage1Error, match="max_tokens"):
        run_stage1(brief, client=fake_client)
