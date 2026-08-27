"""Module 2 (price fetch) unit tests. The staleness guard is a pure
function once "today" is injectable, so it's testable without hitting
yfinance. fetch_price_data's real network path is exercised live against
real tickers as part of the fetch-layer hand-check, not mocked here — but
its NaN-close handling (see test_fetch_price_data_skips_nan_latest_close_*
below) is pure logic over a DataFrame once _fetch_ohlcv_pair is injectable,
so that part is covered."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockbot.fetch import prices
from stockbot.fetch.prices import StaleDataError, _check_staleness, fetch_price_data


def test_same_day_close_is_not_stale():
    _check_staleness(date(2026, 8, 25), today=date(2026, 8, 25))


def test_monday_after_ordinary_weekend_is_not_stale():
    # Friday close, checked the following Monday
    _check_staleness(date(2026, 8, 21), today=date(2026, 8, 24))


def test_within_five_trading_days_is_not_stale():
    _check_staleness(date(2026, 8, 18), today=date(2026, 8, 25))  # 5 business days


def test_beyond_five_trading_days_raises():
    with pytest.raises(StaleDataError):
        _check_staleness(date(2026, 8, 1), today=date(2026, 8, 25))


def _ohlcv(index: pd.DatetimeIndex, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes}, index=index
    )


def test_fetch_price_data_skips_nan_latest_close_and_uses_last_valid_one(monkeypatch):
    # Regression for a real bug found live on PC Jeweller: yfinance
    # returned a real data gap (NaN Close) on the latest row, which used
    # to become current_price_abs unconditionally. The model, given "nan"
    # in its supplied context, fabricated a plausible price instead of
    # refusing — a real, silent violation of "price data is fatal". Dates
    # relative to today (not hardcoded) so this doesn't go stale itself.
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)
    day_before = yesterday - pd.Timedelta(days=1)
    df = _ohlcv(pd.DatetimeIndex([day_before, yesterday, today]), [50.0, 51.0, np.nan])
    monkeypatch.setattr(prices, "_fetch_ohlcv_pair", lambda symbol: (df, df))

    result = fetch_price_data("PCJEWELLER")

    assert result.current_price_abs == 51.0
    assert result.price_date == yesterday.date()


def test_fetch_price_data_raises_when_every_close_is_invalid(monkeypatch):
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)
    df = _ohlcv(pd.DatetimeIndex([yesterday, today]), [np.nan, 0.0])
    monkeypatch.setattr(prices, "_fetch_ohlcv_pair", lambda symbol: (df, df))

    with pytest.raises(ValueError, match="No valid"):
        fetch_price_data("PCJEWELLER")
