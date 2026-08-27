"""Module 2 — technicals. Pure functions over a DataFrame, no network calls.

Operates only on PriceData.ohlcv_adjusted — see fetch/prices.py for why
adjusted (not unadjusted) is the correct series for SMA/RSI/support-
resistance.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stockbot.models import PriceData, Technicals

SWING_WINDOW_DAYS = 3
SUPPORT_RESISTANCE_LOOKBACK_DAYS = 126  # ~6 months of trading days


def sma(closes: pd.Series, window: int) -> float | None:
    if len(closes) < window:
        return None
    value = closes.rolling(window=window).mean().iloc[-1]
    return None if pd.isna(value) else float(value)


def wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothing is mathematically an EWM with alpha=1/period,
    adjust=False — not a simple average of gains/losses over the window."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def find_support_resistance(
    df: pd.DataFrame,
    lookback_days: int = SUPPORT_RESISTANCE_LOOKBACK_DAYS,
    window: int = SWING_WINDOW_DAYS,
) -> tuple[list[float], list[float]]:
    recent = df.tail(lookback_days)
    lows = recent["Low"].to_numpy()
    highs = recent["High"].to_numpy()
    n = len(recent)

    support_levels: set[float] = set()
    resistance_levels: set[float] = set()
    for i in range(window, n - window):
        lo_segment = lows[i - window : i + window + 1]
        hi_segment = highs[i - window : i + window + 1]
        if lows[i] == lo_segment.min():
            support_levels.add(round(float(lows[i]), 2))
        if highs[i] == hi_segment.max():
            resistance_levels.add(round(float(highs[i]), 2))

    return sorted(support_levels), sorted(resistance_levels)


def compute_technicals(price: PriceData) -> Technicals:
    closes = price.ohlcv_adjusted["Close"]
    as_of_date = price.ohlcv_adjusted.index[-1].date()

    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)

    rsi_series = wilder_rsi(closes, 14)
    latest_rsi = rsi_series.iloc[-1]
    rsi14 = None if pd.isna(latest_rsi) else float(latest_rsi)

    support_abs, resistance_abs = find_support_resistance(price.ohlcv_adjusted)

    return Technicals(
        sma50=sma50,
        sma200=sma200,
        rsi14=rsi14,
        support_abs=support_abs,
        resistance_abs=resistance_abs,
        as_of_date=as_of_date,
        source="computed",
        fetched_at=datetime.now(UTC),
    )
