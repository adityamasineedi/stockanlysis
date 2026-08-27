"""bot.py unit tests for the pure formatting functions — no network, no
Telegram bot token needed. Live testing (a real bot round-trip: HTML
rendering in an actual Telegram client, the ⏳ status-message edit, file
attachment delivery) is still owed once TELEGRAM_BOT_TOKEN is available."""

from datetime import date

from stockbot.bot import esc, format_ambiguous_reply, format_verdict_reply
from stockbot.models import AmbiguousMatch, Analysis, TickerInfo, ValidationResult

VALID_VERDICT_JSON = {
    "verdict": "WATCH",
    "current_price_abs": 412.5,
    "price_date": "2026-08-25",
    "buy_zone_abs": [330.0, 355.0],
    # fair_value_*_abs are Python-computed (compute_valuation) and merged
    # into verdict_json by pipeline.py alongside the model's raw
    # valuation_inputs — this fixture models what pipeline.py actually
    # stores, not what the model's JSON block looks like on its own.
    "valuation_inputs": {
        "eps_base": 25.0, "multiple_base": [16.0, 18.0],
        "eps_bear": 20.0, "multiple_bear": [15.0, 17.0],
        "eps_bull": 30.0, "multiple_bull": [18.0, 20.0],
    },
    "fair_value_base_abs": [420.0, 470.0],
    "fair_value_bear_abs": [300.0, 340.0],
    "fair_value_bull_abs": [540.0, 600.0],
    "confidence": 6,
    "risk": "MEDIUM",
    "business_quality": 7,
    "financial_health": 7,
    "management_quality": 6,
    "earnings_quality": "MEDIUM",
    "holding_period": "3-5 years",
    "reasons_buy": ["Strong moat", "Improving margins"],
    "reasons_avoid": ["Regulatory overhang"],
    "biggest_watch": "Q3 margin trend",
}


def _analysis(missing=None) -> Analysis:
    return Analysis(
        ticker="TEST",
        run_date=date(2026, 8, 25),
        verdict_json=VALID_VERDICT_JSON,
        report_md="# full report",
        costs=39.0,
        validation=ValidationResult(True, []),
        missing=missing or [],
    )


def test_esc_escapes_html_special_characters():
    assert esc("<script>&") == "&lt;script&gt;&amp;"


def test_esc_handles_non_string_input():
    assert esc(6) == "6"


def test_format_verdict_reply_contains_key_fields():
    text = format_verdict_reply(_analysis())
    assert "WATCH" in text
    assert "412.5" in text
    assert "330.0" in text and "355.0" in text
    assert "MEDIUM" in text
    assert "Strong moat" in text
    assert "Regulatory overhang" in text
    assert "Q3 margin trend" in text


def test_format_verdict_reply_uses_html_tags_not_markdown():
    text = format_verdict_reply(_analysis())
    assert "<b>" in text
    assert "**" not in text  # not markdown bold


def test_format_verdict_reply_escapes_llm_text_content():
    analysis = _analysis()
    analysis.verdict_json["biggest_watch"] = "<script>alert(1)</script>"
    text = format_verdict_reply(analysis)
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text


def test_format_verdict_reply_appends_missing_warnings():
    text = format_verdict_reply(_analysis(missing=["MISSING: shareholding — fetch failed"]))
    assert "⚠️" in text
    assert "MISSING: shareholding" in text


def test_format_verdict_reply_no_missing_no_warning_lines():
    text = format_verdict_reply(_analysis())
    assert "⚠️" not in text


def test_format_verdict_reply_stays_under_telegram_limit():
    analysis = _analysis()
    analysis.verdict_json["reasons_buy"] = ["x" * 500] * 20  # deliberately huge
    text = format_verdict_reply(analysis)
    assert len(text) <= 4096


def test_format_ambiguous_reply_lists_all_candidates():
    candidates = AmbiguousMatch(
        candidates=[
            TickerInfo("HDFCBANK", "NSE", "HDFC Bank Limited", None),
            TickerInfo("HDFCAMC", "NSE", "HDFC Asset Management Company Limited", None),
        ],
        scores=[100.0, 90.0],
    )
    text = format_ambiguous_reply(candidates)
    assert "HDFCBANK" in text
    assert "HDFCAMC" in text
    assert "<code>" in text
