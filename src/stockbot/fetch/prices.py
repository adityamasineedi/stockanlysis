"""Module 2 — price fetch. Pure yfinance, no parsing of third-party pages.

Fetches BOTH an adjusted and an unadjusted OHLCV series and keeps them
strictly separated: India has frequent bonus issues and stock splits, so
the two diverge materially over a 2-year window. Adjusted feeds SMA/RSI/
support-resistance (analysis/technicals.py); unadjusted feeds current
price and the 52-week high/low, since those are quoted prices a user
would actually see, not split-adjusted ones.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from stockbot.models import PriceData

STALENESS_TRADING_DAYS = 5


class StaleDataError(Exception):
    pass


def _fetch_ohlcv_pair(yf_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ticker = yf.Ticker(yf_symbol)
    adjusted = ticker.history(period="2y", interval="1d", auto_adjust=True)
    unadjusted = ticker.history(period="2y", interval="1d", auto_adjust=False)
    if adjusted.empty or unadjusted.empty:
        return None
    return adjusted, unadjusted


def _check_staleness(last_close_date: date, today: date | None = None) -> None:
    # Business-day (Mon-Fri) count, not calendar-day count, so an ordinary
    # weekend never looks stale. This doesn't know NSE-specific holidays
    # (no market-calendar library is in the project's stack) — a single
    # holiday can add at most ~1-2 uncounted non-trading weekdays, well
    # inside the 5-trading-day cushion, so it's a deliberate, disclosed
    # approximation rather than a precise trading calendar.
    if today is None:
        today = datetime.now().date()  # noqa: DTZ005 - deliberately local calendar date, not UTC
    trading_days_elapsed = int(np.busday_count(last_close_date, today))
    if trading_days_elapsed > STALENESS_TRADING_DAYS:
        raise StaleDataError(
            f"Latest close is {last_close_date.isoformat()}, "
            f"{trading_days_elapsed} business days before today ({today.isoformat()}) "
            f"— exceeds the {STALENESS_TRADING_DAYS}-trading-day staleness guard."
        )


def fetch_price_data(symbol: str) -> PriceData:
    result = None
    used_symbol = None
    for suffix in (".NS", ".BO"):
        candidate = f"{symbol}{suffix}"
        pair = _fetch_ohlcv_pair(candidate)
        if pair is not None:
            result = pair
            used_symbol = candidate
            break

    if result is None:
        raise ValueError(f"No price data found for {symbol!r} on NSE (.NS) or BSE (.BO)")

    adjusted, unadjusted = result

    # Found live on a real PC Jeweller run: yfinance can return a row for
    # the latest date whose Close is NaN (a real data gap on a thinly
    # traded stock, not a network error) — the old code took
    # unadjusted["Close"].iloc[-1] unconditionally, so this NaN silently
    # became current_price_abs. Nothing downstream checked for it: the
    # model, given "₹nan" in its supplied context, fabricated a plausible-
    # looking price instead of refusing — a direct violation of both this
    # module's "price data is fatal" contract and the master prompt's own
    # "never invent numbers" rule. Use the last row with an actual valid
    # (non-NaN, positive) close instead of blindly trusting the last row,
    # and re-anchor the staleness check to THAT row's date — a stale-but-
    # valid close should still trip staleness, which it wouldn't if only
    # the invalid latest row's (fresh-looking) date were checked.
    valid_closes = unadjusted["Close"].dropna()
    valid_closes = valid_closes[valid_closes > 0]
    if valid_closes.empty:
        raise ValueError(
            f"No valid (non-NaN, positive) closing price found for {symbol!r} in the "
            f"fetched OHLCV history — data exists but every close is unusable"
        )
    last_close_date = valid_closes.index[-1].date()
    _check_staleness(last_close_date)

    current_price_abs = float(valid_closes.iloc[-1])

    window_start = unadjusted.index[-1] - pd.Timedelta(days=365)
    week52 = unadjusted.loc[unadjusted.index >= window_start]
    week52_high_abs = float(week52["High"].max())
    week52_low_abs = float(week52["Low"].min())

    return PriceData(
        current_price_abs=current_price_abs,
        price_date=last_close_date,
        ohlcv_adjusted=adjusted,
        ohlcv_unadjusted=unadjusted,
        week52_high_abs=week52_high_abs,
        week52_low_abs=week52_low_abs,
        source=f"yfinance:{used_symbol}",
        fetched_at=datetime.now(UTC),
    )
