"""liquidity.py — ADV turnover helpers."""

from __future__ import annotations

import pandas as pd

from stockbot.liquidity import compute_adv_inr_cr


def test_adv_from_average_volume_and_price():
    adv, avg = compute_adv_inr_cr(
        current_price_abs=100.0,
        average_volume_shares=100_000.0,
    )
    assert adv == 1.0
    assert avg == 100_000.0


def test_adv_from_ohlcv_when_yfinance_volume_missing():
    ohlcv = pd.DataFrame({"Volume": [3000.0, 3000.0, 3000.0, 3000.0, 3000.0]})
    adv, avg = compute_adv_inr_cr(
        current_price_abs=50.0,
        ohlcv=ohlcv,
    )
    assert adv == 0.02
    assert avg == 3000.0
