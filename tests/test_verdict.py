"""Module 9 (Stage 2 verdict) unit tests. Pure-logic pieces (JSON-block
extraction, schema validation, prompt construction) are tested here
without any network call. Live-tested against the real master prompt and
a real brief — that first live run surfaced two real bugs: MAX_TOKENS=8000
(the plan's own explicit value) was still too low for a full 16-section
report plus JSON block once Opus 5's default adaptive thinking is counted
against the same budget, and a truncated/unparseable response skipped
log_call() entirely, silently losing track of a call that had already
been billed. See test_run_stage2_logs_cost_and_raises_clearly_on_truncation
below for the regression test, and the MAX_TOKENS comment in verdict.py
for the story.

v3 migration: VerdictJSON no longer carries fair_value_abs/
bear_fair_value_abs directly — the model supplies valuation_inputs (EPS +
multiple), and compute_valuation() in verdict.py does the multiplication
in Python. See test_compute_valuation_* below."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stockbot.llm.extract import ExtractionResult
from stockbot.llm.verdict import (
    TruncatedResponseError,
    ValuationInputs,
    VerdictParseError,
    build_user_message,
    compute_valuation,
    extract_verdict_json,
    run_stage2,
)
from stockbot.models import Brief, PriceData, ReportText, Technicals, TickerInfo

NOW = datetime.now(UTC)

VALID_JSON_BLOCK = """
Some report prose here.

## FINAL BEGINNER SUMMARY

More prose.

```json
{
  "verdict": "WATCH",
  "current_price_abs": 412.0,
  "price_date": "2026-08-25",
  "buy_zone_abs": [330.0, 355.0],
  "valuation_inputs": {
    "eps_bear": 10.0, "eps_base": 12.0, "eps_bull": 14.0,
    "multiple_bear": [30.0, 32.0],
    "multiple_base": [35.0, 39.0],
    "multiple_bull": [40.0, 44.0]
  },
  "confidence": 6,
  "risk": "MEDIUM",
  "business_quality": 7,
  "financial_health": 7,
  "management_quality": 6,
  "earnings_quality": "MEDIUM",
  "holding_period": "3-5 years",
  "reasons_buy": ["a", "b"],
  "reasons_avoid": ["c"],
  "biggest_watch": "margin compression",
  "missing_data_impact": "no impact on this reasoning",
  "gates_failed": []
}
```
"""


def test_extract_verdict_json_parses_valid_block():
    verdict = extract_verdict_json(VALID_JSON_BLOCK)
    assert verdict.verdict == "WATCH"
    assert verdict.confidence == 6
    assert verdict.buy_zone_abs == (330.0, 355.0)
    assert verdict.price_date == date(2026, 8, 25)


def test_extract_verdict_json_raises_when_no_block_present():
    with pytest.raises(VerdictParseError):
        extract_verdict_json("just prose, no json block anywhere")


def test_extract_verdict_json_raises_on_malformed_json():
    text = "```json\n{not valid json,,,}\n```"
    with pytest.raises(VerdictParseError):
        extract_verdict_json(text)


def test_extract_verdict_json_raises_on_schema_mismatch():
    text = '```json\n{"verdict": "WATCH"}\n```'  # missing every other required field
    with pytest.raises(VerdictParseError):
        extract_verdict_json(text)


def test_extract_verdict_json_takes_last_block_when_multiple_present():
    text = (
        '```json\n{"example": true}\n```\n'
        "some prose\n"
        + VALID_JSON_BLOCK
    )
    verdict = extract_verdict_json(text)
    assert verdict.verdict == "WATCH"


def _minimal_brief() -> Brief:
    df = pd.DataFrame({"Close": [100.0]})
    return Brief(
        ticker=TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co Limited", isin=None),
        price=PriceData(100.0, date(2026, 8, 25), df, df, 120.0, 80.0, "yfinance", NOW),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], date(2026, 8, 25), "computed", NOW),
        financials=None,
        shareholding=None,
        news=None,
        annual_report=ReportText({}, None, None, False, [], "nse_annual_reports", NOW),
        missing=[],
        token_count=0,
        confidence_ceiling=10,
        generated_at=NOW,
    )


def test_build_user_message_includes_hard_injections():
    message = build_user_message(_minimal_brief(), ExtractionResult())
    assert "this pipeline caps at 7/10 maximum" in message
    assert "You do NOT have a web search tool" in message
    assert "Treat them as [FACT]" in message
    assert "<context>" in message
    assert "<price_and_technicals>" in message
    assert "<instruction>" in message
    assert "Analyze:" in message


def test_build_user_message_uses_brief_confidence_ceiling():
    base = _minimal_brief()
    brief = Brief(
        ticker=base.ticker,
        price=base.price,
        technicals=base.technicals,
        financials=base.financials,
        shareholding=base.shareholding,
        news=base.news,
        annual_report=base.annual_report,
        missing=base.missing,
        token_count=base.token_count,
        confidence_ceiling=4,
        generated_at=base.generated_at,
    )
    message = build_user_message(brief, ExtractionResult())
    assert "this pipeline caps at 4/10 maximum" in message


def test_build_user_message_warns_against_inventing_pledge_when_unconfirmed():
    # Regression test: live testing found the model twice stated a pledge
    # percentage that was never confirmed, even with the generic
    # don't-use-general-knowledge injection in place — this pointed,
    # adjacent-to-the-data warning is the fix.
    message = build_user_message(_minimal_brief(), ExtractionResult())  # shareholding=None
    assert "PLEDGE NOTE" in message
    assert "Do NOT" in message
    assert "<pipeline_note>" in message


def test_build_user_message_omits_pledge_warning_when_confirmed():
    import dataclasses

    from stockbot.models import Shareholding

    confirmed = Shareholding(50.0, 12.5, None, None, "Q1", "NSE", NOW)
    brief = dataclasses.replace(_minimal_brief(), shareholding=confirmed)
    message = build_user_message(brief, ExtractionResult())
    assert "PLEDGE NOTE" not in message
    assert "<pipeline_note>" not in message


def test_build_user_message_includes_extraction_summary():
    extraction = ExtractionResult(
        auditor_opinion_type="qualified",
        auditor_concerns=["going concern doubt"],
    )
    message = build_user_message(_minimal_brief(), extraction)
    assert "qualified" in message
    assert "going concern doubt" in message
    assert "<extraction>" in message


def test_build_user_message_includes_retry_feedback_when_present():
    message = build_user_message(
        _minimal_brief(), ExtractionResult(), extra_instruction="§1 confidence exceeded cap of 7"
    )
    assert "<retry_feedback>" in message
    assert "exceeded cap of 7" in message


def test_build_user_message_omits_retry_section_when_absent():
    message = build_user_message(_minimal_brief(), ExtractionResult())
    assert "<retry_feedback>" not in message
    assert "Retry feedback" not in message

def test_run_stage2_logs_cost_and_raises_clearly_on_truncation(monkeypatch, tmp_path):
    # Regression test for two real bugs found on the first live run: (1)
    # MAX_TOKENS=8000 (the plan's own explicit value) truncated a real
    # report before the JSON block, and (2) the resulting parse failure
    # skipped log_call() entirely, silently losing track of an Opus call
    # that had already been billed. TruncatedResponseError (not
    # VerdictParseError) is raised specifically so pipeline.py can treat
    # this as an infrastructure failure that doesn't consume a validation
    # retry — see the v3 migration note in pipeline.py.
    from stockbot.llm import fixtures as fixtures_module

    monkeypatch.setattr(fixtures_module, "FIXTURES_DIR", tmp_path)

    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text="partial report, no json block")]
    fake_response.stop_reason = "max_tokens"
    fake_response.usage.input_tokens = 5000
    fake_response.usage.output_tokens = 8000
    fake_response.usage.cache_read_input_tokens = 0

    fake_client = MagicMock()
    fake_stream_cm = MagicMock()
    fake_stream_cm.__enter__.return_value.get_final_message.return_value = fake_response
    fake_client.messages.stream.return_value = fake_stream_cm

    logged_calls = []
    monkeypatch.setattr(
        "stockbot.llm.client.log_call",
        lambda **kwargs: (logged_calls.append(kwargs), 45.0)[1],
    )

    with pytest.raises(TruncatedResponseError) as exc_info:
        run_stage2(_minimal_brief(), ExtractionResult(), client=fake_client)

    assert len(logged_calls) == 1  # cost logged despite the failure
    assert exc_info.value.cost_inr == 45.0
    assert "max_tokens=32000" in str(exc_info.value)


def test_compute_valuation_multiplies_eps_by_multiple():
    # Schema shape matches Fix 3's own example exactly: a single EPS value
    # times a [low, high] multiple range. (BEL's original live error used a
    # different, now-retired shape — an EPS *range* times a fixed multiple
    # — so this doesn't reproduce that exact arithmetic; it demonstrates
    # the same underlying guarantee: Python's multiplication is what
    # produces the fair-value range now, never the model's own.)
    inputs = ValuationInputs(
        eps_bear=7.6,
        eps_base=9.0,
        eps_bull=10.2,
        multiple_bear=(30.0, 33.0),
        multiple_base=(36.0, 41.0),
        multiple_bull=(46.0, 50.0),
    )
    valuation = compute_valuation(inputs)
    assert valuation.fair_value_bear_abs == pytest.approx((228.0, 250.8))
    assert valuation.fair_value_base_abs == pytest.approx((324.0, 369.0))
    assert valuation.fair_value_bull_abs == pytest.approx((469.2, 510.0))


def test_compute_valuation_handles_negative_eps():
    # Loss-making company (v3's own sector-adaptation carve-out): a
    # negative EPS flips the ordering of eps * multiple, so the result
    # must still come out low < high rather than assuming multiple[0]'s
    # product is always the smaller one.
    inputs = ValuationInputs(
        eps_bear=-5.0,
        eps_base=-2.0,
        eps_bull=1.0,
        multiple_bear=(10.0, 15.0),
        multiple_base=(10.0, 15.0),
        multiple_bull=(10.0, 15.0),
    )
    valuation = compute_valuation(inputs)
    assert valuation.fair_value_bear_abs == (-75.0, -50.0)
    assert valuation.fair_value_base_abs == (-30.0, -20.0)
    assert valuation.fair_value_bull_abs == (10.0, 15.0)


def test_load_master_prompt_prepends_constitution():
    from stockbot.llm.verdict import load_master_prompt

    text = load_master_prompt()
    assert "Quality-First 3–5 Year Portfolio Constitution" in text
    assert "MASTER ANALYSIS PROTOCOL" in text
    assert text.index("Quality-First") < text.index("MASTER ANALYSIS PROTOCOL")


def test_extract_verdict_json_accepts_constitution_fields_and_null_buy_zone():
    block = """
```json
{
  "verdict": "WATCH",
  "current_price_abs": 412.0,
  "price_date": "2026-08-25",
  "buy_zone_abs": null,
  "valuation_inputs": {
    "eps_bear": 10.0, "eps_base": 12.0, "eps_bull": 14.0,
    "multiple_bear": [30.0, 32.0],
    "multiple_base": [35.0, 39.0],
    "multiple_bull": [40.0, 44.0]
  },
  "confidence": 5,
  "risk": "MEDIUM",
  "business_quality": 6,
  "financial_health": 6,
  "management_quality": 6,
  "earnings_quality": "MEDIUM",
  "holding_period": "THESIS RESEARCH REQUIRED",
  "reasons_buy": [],
  "reasons_avoid": ["five-year evidence weak"],
  "biggest_watch": "no durable growth proof",
  "missing_data_impact": "limits conviction",
  "gates_failed": ["five_year_business_test"],
  "five_year_business_test": {
    "answer": "UNCERTAIN",
    "confidence": "LOW",
    "evidence_for": [],
    "evidence_against": ["cyclical risk"]
  },
  "buy_range_allowed": false,
  "add_range_allowed": false,
  "thesis_status": "THESIS_UNDER_REVIEW",
  "anti_chase_flag": false,
  "thesis_invalidation_triggers": ["Revenue declines two annual periods"],
  "profit_review": {"status": "NOT_TRIGGERED", "trigger_reason": [], "note": null},
  "position_building_plan": null
}
```
"""
    verdict = extract_verdict_json(block)
    assert verdict.buy_zone_abs is None
    assert verdict.buy_range_allowed is False
    assert verdict.five_year_business_test is not None
    assert verdict.five_year_business_test.answer == "UNCERTAIN"
