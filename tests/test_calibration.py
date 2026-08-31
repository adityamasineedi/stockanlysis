"""Did the bot's own calls work — pure maths, no I/O."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stockbot.calibration import (
    MIN_DAYS_TO_ANNUALIZE,
    MIN_SAMPLE,
    age_band,
    annualized_pct,
    bucket_by_age,
    bucket_by_label,
    build_bucket,
    days_since,
    forward_return_pct,
    score_calls,
    tier_spread,
)

NOW = datetime(2026, 8, 31, tzinfo=UTC)


def _row(ticker: str, verdict: str, entry: float | None, days_ago: int) -> dict:
    return {
        "ticker": ticker,
        "verdict": verdict,
        "price_at_scan": entry,
        "logged_at": (NOW - timedelta(days=days_ago)).isoformat(),
    }


def _calls(verdict: str, entry: float, now_price: float, days: int, count: int):
    rows = [_row(f"{verdict[:3]}{i}", verdict, entry, days) for i in range(count)]
    prices = {f"{verdict[:3]}{i}".upper(): now_price for i in range(count)}
    return score_calls(rows, prices, now=NOW)


def test_forward_return_and_none_safety():
    assert forward_return_pct(400, 440) == 10.0
    assert forward_return_pct(400, 360) == -10.0
    for bad in (None, 0, -5, float("nan"), float("inf"), "400", True):
        assert forward_return_pct(bad, 440) is None, bad
        assert forward_return_pct(400, bad) is None, bad


def test_days_since_handles_naive_and_bad_timestamps():
    assert days_since((NOW - timedelta(days=30)).isoformat(), NOW) == 30
    # SQLite can hand back a naive timestamp; it must not raise.
    assert days_since("2026-08-01T00:00:00", NOW) == 30
    # A future timestamp is clamped rather than reported negative.
    assert days_since((NOW + timedelta(days=5)).isoformat(), NOW) == 0
    for bad in (None, "", "garbage", 12345):
        assert days_since(bad, NOW) is None, bad


def test_age_bands_are_half_open_and_contiguous():
    assert age_band(0) == "under 3 months"
    assert age_band(89) == "under 3 months"
    assert age_band(90) == "3-12 months"
    assert age_band(364) == "3-12 months"
    assert age_band(365) == "over 12 months"
    assert age_band(-1) is None
    assert age_band(None) is None


def test_short_windows_are_never_annualized():
    """+8% over three weeks annualizes to ~+250% — noise wearing a suit."""
    assert annualized_pct(8.0, 21) is None
    assert annualized_pct(10.0, MIN_DAYS_TO_ANNUALIZE - 1) is None
    assert annualized_pct(21.0, 365) == pytest.approx(21.0)
    # 15% over 200 days: 1.15^(365/200) - 1
    assert annualized_pct(15.0, 200) == pytest.approx(29.05, abs=0.05)
    # A total loss has no defined rate.
    assert annualized_pct(-100.0, 365) is None


def test_bucket_withholds_statistics_below_min_sample():
    """Three data points have no central tendency worth printing, but the
    count is still a fact."""
    few = _calls("AUTO_DEEP_ANALYSIS", 100, 150, 200, MIN_SAMPLE - 1)
    bucket = build_bucket("AUTO", few)
    assert bucket.n == MIN_SAMPLE - 1
    assert bucket.median_return_pct is None
    assert bucket.mean_return_pct is None
    assert bucket.is_reportable is False
    assert bucket.annualized_median_pct is None

    enough = _calls("AUTO_DEEP_ANALYSIS", 100, 150, 200, MIN_SAMPLE)
    reportable = build_bucket("AUTO", enough)
    assert reportable.n == MIN_SAMPLE
    assert reportable.median_return_pct == 50.0
    assert reportable.is_reportable is True

    empty = build_bucket("AUTO", [])
    assert empty.n == 0 and empty.is_reportable is False


def test_unscoreable_rows_are_dropped_not_counted_flat():
    """A call with an unknown outcome is not a zero-return call; counting it
    as one would quietly drag every median toward zero."""
    rows = [
        {"ticker": "A", "verdict": "V"},                                   # no price, no time
        {"ticker": "B", "verdict": "V", "price_at_scan": None, "logged_at": NOW.isoformat()},
        {"ticker": "C", "verdict": "V", "price_at_scan": 100, "logged_at": "garbage"},
        {"ticker": "", "verdict": "V", "price_at_scan": 100, "logged_at": NOW.isoformat()},
        {"ticker": "E", "verdict": "", "price_at_scan": 100, "logged_at": NOW.isoformat()},
    ]
    prices = {"A": 1.0, "B": 1.0, "C": 1.0, "E": 1.0}
    assert score_calls(rows, prices, now=NOW) == []

    # A ticker with no current price is dropped too.
    good = [_row("D", "V", 100, 30)]
    assert score_calls(good, {"D": None}, now=NOW) == []
    assert len(score_calls(good, {"D": 120.0}, now=NOW)) == 1


def test_tier_spread_is_the_headline_and_survives_any_market():
    """Both tiers rode the same market, so the spread needs no benchmark."""
    calls = _calls("AUTO_DEEP_ANALYSIS", 100, 115, 200, 6)
    calls += _calls("NOT_SUITABLE_FOR_3Y_RESEARCH", 100, 102, 200, 6)

    spread = tier_spread(calls, {"AUTO_DEEP_ANALYSIS"}, {"NOT_SUITABLE_FOR_3Y_RESEARCH"})
    assert spread.positive_median_pct == 15.0
    assert spread.negative_median_pct == 2.0
    assert spread.spread_pct == 13.0
    assert spread.discriminates is True
    assert spread.n_positive == 6 and spread.n_negative == 6


def test_a_zero_or_negative_spread_is_a_real_finding():
    """The score not discriminating is the honest answer, not an error — the
    report must be able to say it."""
    calls = _calls("AUTO_DEEP_ANALYSIS", 100, 102, 200, 6)
    calls += _calls("NOT_SUITABLE_FOR_3Y_RESEARCH", 100, 115, 200, 6)

    spread = tier_spread(calls, {"AUTO_DEEP_ANALYSIS"}, {"NOT_SUITABLE_FOR_3Y_RESEARCH"})
    assert spread is not None
    assert spread.spread_pct == -13.0
    assert spread.discriminates is False


def test_tier_spread_refuses_thin_evidence():
    """A spread computed off three rejects is not evidence of anything."""
    calls = _calls("AUTO_DEEP_ANALYSIS", 100, 115, 200, 6)
    calls += _calls("NOT_SUITABLE_FOR_3Y_RESEARCH", 100, 102, 200, MIN_SAMPLE - 1)
    assert tier_spread(calls, {"AUTO_DEEP_ANALYSIS"}, {"NOT_SUITABLE_FOR_3Y_RESEARCH"}) is None
    # Missing side entirely.
    assert tier_spread(calls, {"AUTO_DEEP_ANALYSIS"}, {"NOTHING"}) is None


def test_buckets_group_and_order():
    calls = _calls("AUTO_DEEP_ANALYSIS", 100, 115, 200, 7)
    calls += _calls("HOLDING_MONITOR_ONLY", 100, 105, 200, 5)

    by_label = bucket_by_label(calls)
    assert [b.label for b in by_label] == ["AUTO_DEEP_ANALYSIS", "HOLDING_MONITOR_ONLY"]
    assert [b.n for b in by_label] == [7, 5]  # largest sample first


def test_age_buckets_keep_chronological_order():
    calls = _calls("V", 100, 110, 400, 5)   # over 12 months
    calls += _calls("W", 100, 110, 30, 5)   # under 3 months
    labels = [b.label for b in bucket_by_age(calls)]
    assert labels == ["under 3 months", "over 12 months"]
