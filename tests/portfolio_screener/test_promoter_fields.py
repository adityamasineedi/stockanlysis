"""Promoter holding vs pledge — never confuse the two."""

from __future__ import annotations

from stockbot.portfolio_screener.eligibility import (
    _deterministic_weak_fundamentals_copy,
    _sanitize_promoter_field_confusion,
)
from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.red_flags import collect_red_flags, governance_notes


def test_sanitize_rewrites_promoter_holding_marked_critical():
    raw = "quant_score 30; OCF/PAT 0.37 (Critical); promoter_pct 69.95% (Critical)"
    cleaned = _sanitize_promoter_field_confusion(raw)
    assert "69.95%" in cleaned
    assert "informational" in cleaned.lower()
    assert "promoter_pct 69.95% (Critical)" not in cleaned


def test_high_promoter_holding_never_creates_red_flag():
    m = StockMetrics(
        ticker="BBOX",
        promoter_holding_pct=69.95,
        pledged_promoter_holding_pct=0.0,
        ocf_to_pat=1.0,
        net_income=100.0,
    )
    flags = collect_red_flags(m)
    assert not any("promoter_holding" in f.message.lower() for f in flags)
    assert not any(f.code.startswith("PLEDGE") for f in flags)


def test_high_pledge_is_critical_red_flag():
    m = StockMetrics(
        ticker="X",
        promoter_holding_pct=55.0,
        pledged_promoter_holding_pct=28.4,
    )
    flags = collect_red_flags(m)
    assert any(f.code == "PLEDGE_ELEVATED" for f in flags)
    assert any("pledged_promoter_holding_pct 28.4%" in f.message for f in flags)


def test_governance_notes_separate_holding_and_pledge():
    m = StockMetrics(
        ticker="BBOX",
        promoter_holding_pct=69.95,
        pledged_promoter_holding_pct=None,
    )
    notes = governance_notes(m)
    assert any("promoter_holding_pct 69.95%" in n and "informational" in n for n in notes)
    assert any("pledged_promoter_holding_pct null" in n for n in notes)


def test_deterministic_why_omits_promoter_holding():
    reason, risk = _deterministic_weak_fundamentals_copy(
        quant_score=30.26,
        band="REMOVE",
        ocf_to_pat=0.37,
        red_flag_codes=["OCF_PAT_GAP"],
    )
    assert "OCF/PAT 0.37" in reason
    assert "promoter" not in reason.lower()
    assert "cash conversion" in risk.lower()
