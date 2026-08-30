"""Digest extraction for Telegram — does not change LLM output."""

from stockbot.models import Analysis, ValidationResult
from stockbot.report_digest import (
    build_compact_attachment_md,
    extract_beginner_summary,
)


def test_extract_beginner_summary_stops_before_json():
    report = (
        "### 1. VERDICT\nLong section omitted.\n\n"
        "**SHOULD I BUY?**\n"
        "- **Decision:** WATCH\n"
        "- **In simple words:** Wait for a better price.\n\n"
        "```json\n{\"verdict\": \"WATCH\"}\n```\n"
        "*Research and education, not investment advice.*\n"
    )
    summary = extract_beginner_summary(report)
    assert summary.startswith("SHOULD I BUY?")
    assert "json" not in summary
    assert "Research and education" not in summary


def test_compact_attachment_shorter_than_full_report():
    full = "# Full report\n" + ("Section body.\n" * 500)
    analysis = Analysis(
        ticker="TEST",
        run_date=__import__("datetime").date(2026, 8, 25),
        verdict_json={
            "verdict": "WATCH",
            "risk": "MEDIUM",
            "confidence": 6,
            "holding_period": "3-5 years",
            "reasons_buy": ["Moat"],
            "reasons_avoid": ["Debt"],
            "biggest_watch": "Margins",
        },
        report_md=full
        + "\n\n**SHOULD I BUY?**\n- **Decision:** WATCH\n",
        costs=1.0,
        validation=ValidationResult(True, []),
        missing=["MISSING: pledge"],
    )
    digest = build_compact_attachment_md(analysis)
    assert len(digest) < len(full)
    assert "digest only" in digest.lower()
    assert "MISSING: pledge" in digest
