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


BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULT = 2.0


def bollinger_bands(
    closes: pd.Series,
    window: int = BOLLINGER_WINDOW,
    std_mult: float = BOLLINGER_STD_MULT,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Returns (mid, upper, lower, bandwidth_pct) for the latest bar."""
    if len(closes) < window:
        return None, None, None, None
    rolling = closes.rolling(window=window)
    mid_series = rolling.mean()
    std_series = rolling.std(ddof=0)
    mid = mid_series.iloc[-1]
    std = std_series.iloc[-1]
    if pd.isna(mid) or pd.isna(std):
        return None, None, None, None
    upper = float(mid + std_mult * std)
    lower = float(mid - std_mult * std)
    mid_f = float(mid)
    bandwidth = None
    if mid_f > 0:
        bandwidth = round((upper - lower) / mid_f * 100.0, 2)
    return round(mid_f, 2), round(upper, 2), round(lower, 2), bandwidth


def price_vs_bollinger_label(
    price: float,
    upper: float | None,
    lower: float | None,
) -> str | None:
    if upper is None or lower is None:
        return None
    if price > upper:
        return "above_upper_band"
    if price < lower:
        return "below_lower_band"
    return "inside_bands"


def trend_label(
    price: float,
    sma50: float | None,
    sma200: float | None,
) -> str | None:
    if sma50 is None or sma200 is None:
        return None
    above50 = price >= sma50
    above200 = price >= sma200
    if above50 and above200:
        return "uptrend"
    if not above50 and not above200:
        return "downtrend"
    return "mixed"


def compute_technicals(price: PriceData) -> Technicals:
    closes = price.ohlcv_adjusted["Close"]
    as_of_date = price.ohlcv_adjusted.index[-1].date()

    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)

    rsi_series = wilder_rsi(closes, 14)
    latest_rsi = rsi_series.iloc[-1]
    rsi14 = None if pd.isna(latest_rsi) else float(latest_rsi)

    support_abs, resistance_abs = find_support_resistance(price.ohlcv_adjusted)

    bb_mid, bb_upper, bb_lower, bb_bw = bollinger_bands(closes)
    current_price = float(closes.iloc[-1])
    bb_position = price_vs_bollinger_label(current_price, bb_upper, bb_lower)
    trend = trend_label(current_price, sma50, sma200)

    return Technicals(
        sma50=sma50,
        sma200=sma200,
        rsi14=rsi14,
        support_abs=support_abs,
        resistance_abs=resistance_abs,
        as_of_date=as_of_date,
        source="computed",
        fetched_at=datetime.now(UTC),
        bollinger_mid=bb_mid,
        bollinger_upper=bb_upper,
        bollinger_lower=bb_lower,
        bollinger_bandwidth_pct=bb_bw,
        price_vs_bollinger=bb_position,
        trend_label=trend,
    )
