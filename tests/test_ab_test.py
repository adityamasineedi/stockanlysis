"""ab_test.py unit tests — the pure comparison/scoring logic only. No
network, no API keys needed. The actual DeepSeek call (_call_deepseek) is
tested only for its "no API key configured" early-return path here; a
real network call is out of scope for the unit suite by the project's own
"nothing testable for free should cost one" rule."""


from datetime import date

from stockbot.ab_test import (
    ExtractionAttempt,
    check_critical_findings,
    format_comparison_table,
)
from stockbot.llm.extract import ExtractionResult
from stockbot.models import RedFlag


def test_check_critical_findings_all_present():
    result = ExtractionResult(
        auditor_opinion_type="clean",
        auditor_concerns=[
            "Rule 11(g) audit-trail feature was not enabled for part of the year",
            "CARO clause (ii)(b): stock statements filed with banks showed variances against books",
        ],
        key_audit_matters=["Whistle-blower complaints were received during the year and investigated"],
    )
    findings = check_critical_findings(result)
    assert findings == {
        "rule_11g_audit_trail": True,
        "caro_bank_variance": True,
        "whistleblower_complaints": True,
    }


def test_check_critical_findings_all_missing():
    result = ExtractionResult(auditor_opinion_type="clean")
    findings = check_critical_findings(result)
    assert findings == {
        "rule_11g_audit_trail": False,
        "caro_bank_variance": False,
        "whistleblower_complaints": False,
    }


def test_check_critical_findings_caro_requires_bank_context():
    # "CARO" alone (e.g. a routine mention of the report existing) must not
    # count as finding the specific bank-stock-statement variance.
    result = ExtractionResult(auditor_concerns=["CARO report was issued alongside the standalone financials"])
    findings = check_critical_findings(result)
    assert findings["caro_bank_variance"] is False


def test_check_critical_findings_searches_multiple_fields():
    # A real extraction might place a finding under related_party_summary
    # or extraction_gaps rather than auditor_concerns/key_audit_matters —
    # the check must not only look in one field.
    result = ExtractionResult(
        related_party_summary="unclear; extraction_gaps notes the whistle-blower matter separately",
        extraction_gaps=["Rule 11(g) exception applied to part of the audit trail"],
    )
    findings = check_critical_findings(result)
    assert findings["rule_11g_audit_trail"] is True
    assert findings["whistleblower_complaints"] is True


def test_format_comparison_table_includes_all_models_and_fields():
    attempts = [
        ExtractionAttempt(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            result=ExtractionResult(auditor_opinion_type="clean"),
        ),
        ExtractionAttempt(provider="deepseek", model="deepseek-v4-flash", result=None, error="no key"),
    ]
    table = format_comparison_table(attempts)
    assert "anthropic/claude-haiku-4-5-20251001" in table
    assert "deepseek/deepseek-v4-flash" in table
    assert "auditor_opinion_type" in table
    assert "ERROR: no key" in table
    assert "clean" in table


def test_format_comparison_table_truncates_long_values():
    long_flag = RedFlag(headline="x" * 500, url="http://example.com", published_date=date(2026, 1, 1), found_by_query="q")
    attempts = [
        ExtractionAttempt(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            result=ExtractionResult(red_flags_found=[long_flag]),
        )
    ]
    table = format_comparison_table(attempts)
    # every raw table cell should be bounded, not a raw 500+ char dump
    for line in table.splitlines():
        assert len(line) < 600


def test_deepseek_call_reports_missing_api_key(monkeypatch):
    from stockbot import ab_test

    # Patch where _call_deepseek reads settings (stale if config was reloaded).
    monkeypatch.setattr(ab_test.settings, "deepseek_api_key", "")
    attempt = ab_test._call_deepseek("some user message", "deepseek-v4-flash")
    assert attempt.result is None
    assert "DEEPSEEK_API_KEY" in attempt.error


def test_run_extraction_ab_raises_on_unknown_provider():
    from datetime import UTC, datetime

    import pandas as pd
    import pytest

    from stockbot.ab_test import run_extraction_ab
    from stockbot.models import Brief, PriceData, ReportText, Technicals, TickerInfo

    now = datetime.now(UTC)
    df = pd.DataFrame({"Close": [100.0]})
    brief = Brief(
        ticker=TickerInfo(symbol="TEST", exchange="NSE", company_name="Test Co", isin=None),
        price=PriceData(100.0, date(2026, 1, 1), df, df, 120.0, 80.0, "yfinance", now),
        technicals=Technicals(95.0, 90.0, 55.0, [85.0], [110.0], date(2026, 1, 1), "computed", now),
        financials=None, shareholding=None, news=None,
        annual_report=ReportText({}, None, None, False, [], "nse_annual_reports", now),
        missing=[], token_count=0, confidence_ceiling=10, generated_at=now,
    )
    with pytest.raises(ValueError, match="Unknown provider"):
        run_extraction_ab(brief, [("openai", "gpt-4")])
