"""SIP (Systematic Investment Plan) maths — pure functions, no I/O.

Scope note: this bot's universe is NSE equities (``fetch/tickers.py`` loads
EQUITY_L.csv), so a plan here is a monthly contribution into an individual
stock, not into a mutual fund. Two consequences the callers must surface
rather than hide:

1. The familiar "large-cap 10-12%, small-cap 14-18%" bands are *fund* figures.
   Single stocks disperse far wider and some go to zero, so those rates are
   only ever assumptions the user chose — never a forecast this module
   endorses. Prefer a stock's own valuation-derived scenarios when one exists.
2. Dip top-ups here add on price decline alone. That is a deliberate,
   documented exception to the portfolio constitution's "a falling price is
   never a reason to add" (principle 2), scoped to declared SIP plans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

DipSeverity = Literal["MODERATE", "DEEP"]

# Spec thresholds, measured against the recent 3-month high.
MODERATE_DIP_PCT = 5.0
DEEP_DIP_PCT = 10.0

# Top-up multiples of the normal monthly amount, per the spec.
_TOPUP_MULTIPLES: dict[str, tuple[float, float]] = {
    "MODERATE": (0.5, 1.0),
    "DEEP": (1.0, 2.0),
}

THREE_MONTH_TRADING_DAYS = 63

DEFAULT_HORIZON_YEARS = 20
# Only used when the ticker has no stored analysis to derive scenarios from.
DEFAULT_SCENARIO_RATES_PCT = (10.0, 12.0, 14.0)


@dataclass(frozen=True)
class SipProjection:
    """One scenario: what goes in, and what it might become."""

    annual_rate_pct: float
    years: int
    total_invested: float
    projected_corpus: float

    @property
    def gain(self) -> float:
        return self.projected_corpus - self.total_invested


def project_corpus(
    monthly_amount: float,
    years: int,
    annual_rate_pct: float,
    *,
    step_up_pct: float = 0.0,
) -> SipProjection:
    """Future value of a monthly SIP, compounded monthly.

    ``step_up_pct`` raises the contribution on each anniversary (a 10% step-up
    on 5,000 pays 5,500 through year 2), which is how step-up SIPs are actually
    sold — the raise applies to the instalment, not retroactively.

    A negative ``annual_rate_pct`` is allowed: a plan can lose money, and
    refusing to model that would only flatter the projection. Rates at or below
    -100%/yr are rejected as meaningless rather than returned as a complex or
    zero corpus.
    """
    if monthly_amount <= 0 or years <= 0:
        return SipProjection(annual_rate_pct, max(years, 0), 0.0, 0.0)
    if annual_rate_pct <= -100.0:
        raise ValueError(f"annual_rate_pct must be > -100, got {annual_rate_pct!r}")

    monthly_rate = (1.0 + annual_rate_pct / 100.0) ** (1.0 / 12.0) - 1.0
    months = years * 12

    total_invested = 0.0
    corpus = 0.0
    for month in range(months):
        instalment = monthly_amount * ((1.0 + step_up_pct / 100.0) ** (month // 12))
        # Contribute first, then let the whole balance compound for the month.
        corpus = (corpus + instalment) * (1.0 + monthly_rate)
        total_invested += instalment

    return SipProjection(
        annual_rate_pct=annual_rate_pct,
        years=years,
        total_invested=round(total_invested, 2),
        projected_corpus=round(corpus, 2),
    )


def next_step_up_amount(
    monthly_amount: float, step_up_pct: float, elapsed_years: int
) -> float:
    """The instalment due after ``elapsed_years`` anniversaries."""
    if monthly_amount <= 0 or elapsed_years < 0:
        return max(monthly_amount, 0.0)
    return round(monthly_amount * ((1.0 + step_up_pct / 100.0) ** elapsed_years), 2)


def three_month_high(ohlcv: pd.DataFrame) -> float | None:
    """Highest close over roughly the last 3 months of daily bars.

    Takes the frame already on ``PriceData.ohlcv_adjusted`` so the fetch layer
    is untouched. NaN closes are dropped first: yfinance really does return
    rows with a NaN Close on thinly traded names (see the comment in
    ``fetch/prices.py``), and ``max()`` over them would poison the dip maths.
    Returns None rather than a guess when there is nothing usable.
    """
    if ohlcv is None or len(ohlcv) == 0 or "Close" not in ohlcv:
        return None
    closes = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return None
    high = float(closes.tail(THREE_MONTH_TRADING_DAYS).max())
    return high if math.isfinite(high) and high > 0 else None


def dip_pct_from_high(current_price: float, high_3m: float | None) -> float | None:
    """How far below the 3-month high the price sits, as a positive percent."""
    if high_3m is None or high_3m <= 0 or current_price <= 0:
        return None
    if current_price >= high_3m:
        return 0.0
    return round((1.0 - current_price / high_3m) * 100.0, 2)


def classify_dip(current_price: float, high_3m: float | None) -> DipSeverity | None:
    """None | MODERATE (5-10% off the high) | DEEP (>10% off).

    Boundaries: exactly 5% is MODERATE, exactly 10% is still MODERATE, and
    anything past 10% is DEEP — the spec reads "5-10%" and ">10%", so 10.0
    belongs to the closed lower band.
    """
    drop = dip_pct_from_high(current_price, high_3m)
    if drop is None or drop < MODERATE_DIP_PCT:
        return None
    return "MODERATE" if drop <= DEEP_DIP_PCT else "DEEP"


def suggest_topup(dip: DipSeverity | None, monthly_amount: float) -> tuple[float, float] | None:
    """Rupee range for a one-time top-up, or None when no dip is live.

    Deliberately a range, never a single number — the caller must not present
    this as a computed "correct" amount.
    """
    if dip is None or monthly_amount <= 0:
        return None
    low_mult, high_mult = _TOPUP_MULTIPLES[dip]
    return (round(monthly_amount * low_mult, 2), round(monthly_amount * high_mult, 2))


def scenario_projections(
    monthly_amount: float,
    years: int,
    rates_pct: tuple[float, ...],
    *,
    step_up_pct: float = 0.0,
) -> list[SipProjection]:
    """One projection per supplied rate, ordered as given (worst → best)."""
    return [
        project_corpus(monthly_amount, years, rate, step_up_pct=step_up_pct)
        for rate in rates_pct
    ]
