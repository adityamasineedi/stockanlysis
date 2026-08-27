"""pipeline.py orchestration tests. Every fetch/LLM/storage call is
monkeypatched — no network, no real API key needed, no real DB touched.
Live end-to-end testing (a real run_full_analysis against real APIs) is
still owed once ANTHROPIC_API_KEY is available."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stockbot import pipeline as pipeline_module
from stockbot.llm.extract import ExtractionResult
from stockbot.models import (
    AmbiguousMatch,
    Brief,
    PriceData,
    ReportText,
    Technicals,
    TickerInfo,
    ValidationResult,
)
from stockbot.pipeline import run_full_analysis

NOW = datetime.now(UTC)
TICKER = TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co Limited", isin=None)

PASSING_VERDICT_JSON = """```json
{
  "verdict": "WATCH", "current_price_abs": 400.0, "price_date": "PRICE_DATE_PLACEHOLDER",
  "buy_zone_abs": [370.0, 380.0],
  "valuation_inputs": {
    "eps_base": 25.0, "multiple_base": [16.0, 18.0],
    "eps_bear": 20.0, "multiple_bear": [15.0, 17.0],
    "eps_bull": 30.0, "multiple_bull": [18.0, 20.0]
  },
  "confidence": 6, "risk": "LOW",
  "business_quality": 7, "financial_health": 7, "management_quality": 7,
  "earnings_quality": "HIGH", "holding_period": "3-5 years",
  "reasons_buy": ["a"], "reasons_avoid": ["b"], "biggest_watch": "c",
  "missing_data_impact": "no meaningful impact", "gates_failed": []
}
```""".replace("PRICE_DATE_PLACEHOLDER", datetime.now(UTC).date().isoformat())


def _brief() -> Brief:
    df = pd.DataFrame({"Close": [100.0]})
    return Brief(
        ticker=TICKER,
        price=PriceData(100.0, date(2026, 8, 25), df, df, 120.0, 80.0, "yfinance", NOW),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], date(2026, 8, 25), "computed", NOW),
        financials=None,
        shareholding=None,
        news=None,
        annual_report=ReportText({}, None, None, False, [], "nse_annual_reports", NOW),
        missing=[],
        token_count=100,
        confidence_ceiling=10,
        generated_at=NOW,
    )


def _patch_common(monkeypatch, *, resolve_result=TICKER, cached=None, budget_ok=True, spent=0.0):
    monkeypatch.setattr(pipeline_module, "load_symbol_table", lambda: object())
    monkeypatch.setattr(pipeline_module, "resolve_ticker", lambda query, table: resolve_result)
    monkeypatch.setattr(pipeline_module.storage, "get_cached", lambda ticker, max_age_days=7: cached)
    monkeypatch.setattr(pipeline_module, "check_budget", lambda: (budget_ok, spent))
    monkeypatch.setattr(pipeline_module, "assemble_brief", lambda ticker: _brief())
    monkeypatch.setattr(
        pipeline_module, "run_stage1", lambda brief: (ExtractionResult(), {"input_tokens": 100, "output_tokens": 50, "cost_inr": 5.0})
    )
    monkeypatch.setattr(pipeline_module.storage, "save_analysis", lambda **kwargs: 1)


def test_not_found_short_circuits(monkeypatch):
    _patch_common(monkeypatch, resolve_result=None)
    result = run_full_analysis("nonsense query")
    assert result.status == "not_found"


def test_ambiguous_short_circuits(monkeypatch):
    ambiguous = AmbiguousMatch(candidates=[TICKER], scores=[80.0])
    _patch_common(monkeypatch, resolve_result=ambiguous)
    result = run_full_analysis("hdfc")
    assert result.status == "ambiguous"
    assert result.candidates is ambiguous


def test_cache_hit_skips_budget_and_llm_entirely(monkeypatch):
    from stockbot.models import Analysis

    cached_analysis = Analysis(
        ticker="TEST", run_date=date(2026, 8, 19), verdict_json={}, report_md="# cached",
        costs=39.0, validation=ValidationResult(True, []), missing=[],
    )
    _patch_common(monkeypatch, cached=cached_analysis)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not be called on a cache hit")

    monkeypatch.setattr(pipeline_module, "check_budget", _fail_if_called)
    monkeypatch.setattr(pipeline_module, "assemble_brief", _fail_if_called)

    result = run_full_analysis("TEST")
    assert result.status == "ok"
    assert result.analysis is cached_analysis


def test_budget_exceeded_short_circuits_before_any_llm_call(monkeypatch):
    _patch_common(monkeypatch, budget_ok=False, spent=1450.0)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not reach the brief/LLM stage")

    monkeypatch.setattr(pipeline_module, "assemble_brief", _fail_if_called)

    result = run_full_analysis("TEST")
    assert result.status == "budget_exceeded"
    assert result.spent_inr == 1450.0


def test_successful_run_saves_and_returns_ok(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        pipeline_module,
        "run_stage2",
        lambda brief, extraction, extra_instruction=None: (
            PASSING_VERDICT_JSON,
            None,
            {"input_tokens": 200, "output_tokens": 100, "cost_inr": 20.0},
        ),
    )

    result = run_full_analysis("TEST")
    assert result.status == "ok"
    assert result.analysis is not None
    assert result.analysis.costs == pytest.approx(25.0)  # stage1 5.0 + stage2 20.0


def test_validation_failure_retries_once_then_succeeds(monkeypatch):
    _patch_common(monkeypatch)
    calls = {"count": 0}

    def _stage2(brief, extraction, extra_instruction=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return "```json\n{\"verdict\": \"WATCH\"}\n```", None, {
                "input_tokens": 100, "output_tokens": 50, "cost_inr": 10.0
            }
        return PASSING_VERDICT_JSON, None, {"input_tokens": 100, "output_tokens": 50, "cost_inr": 10.0}

    monkeypatch.setattr(pipeline_module, "run_stage2", _stage2)

    result = run_full_analysis("TEST")
    assert result.status == "ok"
    assert calls["count"] == 2


def test_validation_failure_twice_returns_insufficient_data(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        pipeline_module,
        "run_stage2",
        lambda brief, extraction, extra_instruction=None: (
            "```json\n{\"verdict\": \"WATCH\"}\n```",
            None,
            {"input_tokens": 100, "output_tokens": 50, "cost_inr": 10.0},
        ),
    )

    result = run_full_analysis("TEST")
    assert result.status == "insufficient_data"
    assert result.validation_failures
    assert result.analysis is None


def test_per_analysis_cost_cap_stops_before_stage2_when_stage1_alone_exceeds_it(monkeypatch):
    # Regression test for the real ₹243 overnight run: nothing was
    # stopping repeated expensive calls on one analysis. This asserts
    # Stage 2 (and any retry) never even gets called once the cap is hit.
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        pipeline_module,
        "run_stage1",
        lambda brief: (ExtractionResult(), {"input_tokens": 1, "output_tokens": 1, "cost_inr": 90.0}),
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Stage 2 must not be called once the per-analysis cap is exceeded")

    monkeypatch.setattr(pipeline_module, "run_stage2", _fail_if_called)

    result = run_full_analysis("TEST")
    assert result.status == "analysis_cost_exceeded"
    assert result.spent_inr == 90.0


def test_per_analysis_cost_cap_stops_after_stage2_first_attempt(monkeypatch):
    _patch_common(monkeypatch)  # stage1 cost_inr=5.0 from _patch_common's default
    monkeypatch.setattr(
        pipeline_module,
        "run_stage2",
        lambda brief, extraction, extra_instruction=None: (
            "```json\n{\"verdict\": \"WATCH\"}\n```",  # fails validation too, but cost cap wins first
            None,
            {"input_tokens": 1, "output_tokens": 1, "cost_inr": 76.0},  # 5 + 76 = 81 > 80
        ),
    )

    result = run_full_analysis("TEST")
    assert result.status == "analysis_cost_exceeded"
    assert result.spent_inr == pytest.approx(81.0)


def test_per_analysis_cost_cap_stops_after_a_retry_pushes_it_over(monkeypatch):
    _patch_common(monkeypatch)  # stage1 cost_inr=5.0
    calls = {"count": 0}

    def _stage2(brief, extraction, extra_instruction=None):
        calls["count"] += 1
        # always fails validation (bare-bones JSON) and costs 40 each time:
        # attempt 1 -> running 5+40=45 (under cap, retries);
        # attempt 2 (retry) -> running 45+40=85 (over cap, must stop here)
        return (
            '```json\n{"verdict": "WATCH"}\n```',
            None,
            {"input_tokens": 1, "output_tokens": 1, "cost_inr": 40.0},
        )

    monkeypatch.setattr(pipeline_module, "run_stage2", _stage2)

    result = run_full_analysis("TEST")
    assert result.status == "analysis_cost_exceeded"
    assert calls["count"] == 2
    assert result.spent_inr == pytest.approx(85.0)
