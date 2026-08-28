"""Tests for single-ticker eligibility + telegram plain-text shortcut."""

from __future__ import annotations

from stockbot.bot import _parse_prescan_plain_text
from stockbot.portfolio_screener.eligibility import (
    EligibilityResult,
    _normalize_verdict,
    _verdict_from_band,
)
from stockbot.portfolio_screener.issuer_routing import WC_RECONCILIATION_CHECKLIST


def test_verdict_bands():
    assert _verdict_from_band("STRONG_CANDIDATE", "PASS") == (
        "AUTO_DEEP_ANALYSIS",
        True,
    )
    assert _verdict_from_band("WATCHLIST", "PASS") == (
        "SECTOR_SPECIFIC_REVIEW",
        True,
    )
    assert _verdict_from_band("REMOVE", "PASS") == ("HOLDING_MONITOR_ONLY", False)
    assert _verdict_from_band("STRONG_CANDIDATE", "HARD_EXCLUDE") == (
        "NOT_SUITABLE_FOR_3Y_RESEARCH",
        False,
    )
    assert _verdict_from_band("REMOVE", "DATA_UNAVAILABLE") == (
        "DATA_UNAVAILABLE_RETRY",
        False,
    )


def test_normalize_legacy_verdicts():
    assert _normalize_verdict("SUITABLE_FOR_DEEP_ANALYSIS") == "AUTO_DEEP_ANALYSIS"
    assert _normalize_verdict("REVIEW_EXCEPTION") == "SECTOR_SPECIFIC_REVIEW"
    assert _normalize_verdict("MODEL_NOT_APPLICABLE") == "SECTOR_SPECIFIC_REVIEW"
    assert _normalize_verdict("DATA_UNAVAILABLE") == "DATA_UNAVAILABLE_RETRY"
    assert _normalize_verdict("NOT_SUITABLE") == "NOT_SUITABLE_FOR_3Y_RESEARCH"
    assert _normalize_verdict("AUTO_DEEP_ANALYSIS") == "AUTO_DEEP_ANALYSIS"


def test_parse_prescan_plain_text():
    assert _parse_prescan_plain_text("prescan BEL") == "BEL"
    assert _parse_prescan_plain_text("Pre-scan  TCS") == "TCS"
    assert _parse_prescan_plain_text("prescreen INFY") == "INFY"
    assert _parse_prescan_plain_text("analyze BEL") is None
    assert _parse_prescan_plain_text("prescan") is None


def test_cheap_wc_telegram_card_wording_and_checklist():
    result = EligibilityResult(
        query="MAZDOCK",
        ticker="MAZDOCK",
        company_name="Mazagon Dock Shipbuilders Limited",
        verdict="SECTOR_SPECIFIC_REVIEW",
        suitable_for_deep_analysis=True,
        final_score=54.8,
        candidate_band="REMOVE",
        quality_score=75.0,
        growth_score=83.0,
        financial_strength_score=81.0,
        issuer_class="DEFENCE_EPC_PROJECT",
        eligibility_route="DEFENCE_WC_REVIEW",
        cash_conversion_status="ESCALATED_WATCH",
        cash_conversion_interpretation=(
            "Reported cash conversion is extremely weak. This may reflect "
            "milestone billing and project working-capital timing."
        ),
        ocf_pat_current=-0.93,
        ocf_pat_3y_cumulative=0.02,
        cfo_3y_sum_abs=132.0,
        pat_3y_sum_abs=7846.0,
        debt_equity=0.05,
        interest_coverage=37.81,
        net_debt_ebitda=-4.74,
        next_research_action="CHEAP_WC_RECONCILIATION_FIRST",
        quality_override=True,
        key_reason="Strong quality/growth/strength conflict with low generic score.",
        key_risk="Three-year CFO/PAT is 0.02; working-capital explanation is required.",
        computed_metric_warnings=[
            "D/E 0.05 — derived from P&L/BS/CF (not a Screener ratio row)",
        ],
    )
    html = result.telegram_html()
    assert "reported cash conversion is extremely weak" in html.lower()
    assert "High quality, but cash conversion is extremely weak" not in html
    assert "Why routed:" in html
    assert "Why blocked:" in html
    assert "Three-year CFO/PAT is 0.02" in html
    assert "TEMPORARY_BILLING_CYCLE" in html
    assert "WORKING_CAPITAL_STRESS" in html
    assert "Working-capital reconciliation required" in html
    assert str(len(WC_RECONCILIATION_CHECKLIST))  # checklist present
    for item in WC_RECONCILIATION_CHECKLIST[:2]:
        assert item[:40] in html or "Verify CFO and PAT" in html
    assert "₹132 Cr" in html
    assert "₹7846 Cr" in html or "₹7,846" in html or "7846" in html
    assert "D/E: 0.05" in html
