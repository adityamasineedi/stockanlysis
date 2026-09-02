"""Tests for post-analyze long-term ranking."""

from __future__ import annotations

from stockbot.analysis_rank import (
    format_rank_telegram,
    rank_analyses,
    rank_from_verdict,
)


def _verdict(
    *,
    verdict: str = "BUY",
    base: tuple[float, float] = (10.0, 14.0),
    bull: tuple[float, float] = (18.0, 22.0),
    bear: tuple[float, float] = (-5.0, 0.0),
    five: str = "YES",
    buy_allowed: bool = True,
    price: float = 100.0,
    buy_zone: tuple[float, float] = (80.0, 95.0),
    confidence: int = 7,
) -> dict:
    return {
        "verdict": verdict,
        "confidence": confidence,
        "risk": "MEDIUM",
        "current_price_abs": price,
        "buy_zone_abs": list(buy_zone),
        "buy_range_allowed": buy_allowed,
        "five_year_business_test": {"answer": five},
        "thesis_status": "THESIS_CONFIRMING",
        "expected_return": {
            "horizon_years": 3,
            "bear_cagr_range_pct": list(bear),
            "base_cagr_range_pct": list(base),
            "bull_cagr_range_pct": list(bull),
        },
    }


def test_hero_style_correction_ranks_below_strong_buy() -> None:
    hero = rank_from_verdict(
        "HEROMOTOCO",
        _verdict(
            verdict="BUY ON CORRECTION",
            base=(-0.5, 3.1),
            bull=(11.0, 14.0),
            price=5300.0,
            buy_zone=(4130.0, 4410.0),
            buy_allowed=True,
            confidence=6,
        ),
    )
    strong = rank_from_verdict(
        "CAMS",
        _verdict(
            verdict="BUY",
            base=(11.0, 15.0),
            bull=(18.0, 22.0),
            price=90.0,
            buy_zone=(80.0, 95.0),
            confidence=7,
        ),
    )
    assert strong.entry_ready is True
    assert hero.wait_dip is True
    assert strong.score > hero.score


def test_rank_entry_mode_puts_ready_first() -> None:
    rows = [
        (
            "WAIT",
            _verdict(
                verdict="BUY ON CORRECTION",
                base=(12.0, 14.0),
                price=120.0,
                buy_zone=(80.0, 90.0),
            ),
            None,
        ),
        (
            "READY",
            _verdict(verdict="BUY", base=(8.0, 10.0), price=90.0, buy_zone=(80.0, 95.0)),
            None,
        ),
    ]
    ranked = rank_analyses(rows, mode="entry")
    assert ranked[0].ticker == "READY"
    assert ranked[0].entry_ready is True


def test_five_year_no_is_skipped() -> None:
    row = rank_from_verdict("BAD", _verdict(five="NO", base=(20.0, 25.0)))
    assert row.skip_reason is not None
    assert row.score < 0


def test_format_rank_empty() -> None:
    text = format_rank_telegram([], mode="hold", total_analyzed=0)
    assert "/analyze" in text
    assert "No stored" in text
