"""Module 2 (technicals) unit tests. Pure functions, synthetic data, no
network — real-data cross-checks (e.g. RELIANCE against a charting
platform) happen by hand per the module's verification step."""

import numpy as np
import pandas as pd
import pytest

from stockbot.analysis.technicals import (
    bollinger_bands,
    find_support_resistance,
    price_vs_bollinger_label,
    sma,
    trend_label,
    wilder_rsi,
)


def test_sma_insufficient_data_returns_none():
    closes = pd.Series([1.0, 2.0, 3.0])
    assert sma(closes, 50) is None


def test_sma_matches_manual_calculation():
    closes = pd.Series([float(i) for i in range(1, 11)])  # 1..10
    # SMA(5) of the last 5 values (6,7,8,9,10) = 8.0
    assert sma(closes, 5) == pytest.approx(8.0)


def test_wilder_rsi_all_gains_is_100():
    closes = pd.Series([float(i) for i in range(1, 31)])  # strictly rising
    rsi = wilder_rsi(closes, period=14)
    assert rsi.iloc[-1] == pytest.approx(100.0)


def test_wilder_rsi_all_losses_is_0():
    closes = pd.Series([float(i) for i in range(30, 0, -1)])  # strictly falling
    rsi = wilder_rsi(closes, period=14)
    assert rsi.iloc[-1] == pytest.approx(0.0)


def test_wilder_rsi_flat_series_is_bounded():
    closes = pd.Series([100.0] * 30)
    rsi = wilder_rsi(closes, period=14)
    # no gains and no losses -> 0/0 -> NaN is the honest answer, not a guess
    assert pd.isna(rsi.iloc[-1])


def test_wilder_rsi_stays_within_0_100_for_random_walk():
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=0.0, scale=1.0, size=200)
    closes = pd.Series(100 + np.cumsum(steps))
    rsi = wilder_rsi(closes, period=14).dropna()
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def _synthetic_ohlcv(lows: list[float], highs: list[float]) -> pd.DataFrame:
    n = len(lows)
    return pd.DataFrame(
        {
            "Low": lows,
            "High": highs,
            "Close": [(low + high) / 2 for low, high in zip(lows, highs)],
        },
        index=pd.date_range("2026-01-01", periods=n, freq="D"),
    )


def test_find_support_resistance_detects_obvious_swing_points():
    # a clean V-shape dip at index 10 (window=3) should register as a swing low
    lows = [100 - abs(i - 10) * 0.5 if i != 10 else 90.0 for i in range(20)]
    highs = [v + 5 for v in lows]
    df = _synthetic_ohlcv(lows, highs)

    support, resistance = find_support_resistance(df, lookback_days=20, window=3)

    assert 90.0 in support
    assert all(isinstance(v, float) for v in support + resistance)
    assert support == sorted(support)
    assert resistance == sorted(resistance)


def test_find_support_resistance_empty_on_short_series():
    df = _synthetic_ohlcv([10.0, 11.0], [12.0, 13.0])
    support, resistance = find_support_resistance(df, lookback_days=20, window=3)
    assert support == []
    assert resistance == []


def test_bollinger_bands_on_flat_series():
    closes = pd.Series([100.0] * 30)
    mid, upper, lower, bw = bollinger_bands(closes, window=20)
    assert mid == pytest.approx(100.0)
    assert upper == pytest.approx(100.0)
    assert lower == pytest.approx(100.0)
    assert bw == pytest.approx(0.0)


def test_price_vs_bollinger_labels():
    assert price_vs_bollinger_label(110.0, 105.0, 95.0) == "above_upper_band"
    assert price_vs_bollinger_label(90.0, 105.0, 95.0) == "below_lower_band"
    assert price_vs_bollinger_label(100.0, 105.0, 95.0) == "inside_bands"


def test_trend_label_from_smas():
    assert trend_label(120.0, 110.0, 100.0) == "uptrend"
    assert trend_label(90.0, 110.0, 100.0) == "downtrend"
    assert trend_label(105.0, 110.0, 100.0) == "mixed"
