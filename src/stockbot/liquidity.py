"""Average daily volume (ADV) — turnover in ₹ crore for liquidity gates."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd


def _round_cr(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_adv_inr_cr(
    *,
    current_price_abs: float | None,
    average_volume_shares: float | None = None,
    ohlcv: pd.DataFrame | None = None,
    lookback_days: int = 20,
) -> tuple[float | None, float | None]:
    """Return (adv_inr_cr, avg_volume_shares). ADV = avg daily turnover in ₹ crore."""
    if current_price_abs is None or current_price_abs <= 0:
        return None, None

    avg_shares = average_volume_shares
    if avg_shares is None and ohlcv is not None and "Volume" in ohlcv.columns:
        volumes = ohlcv["Volume"].dropna()
        if len(volumes) >= 5:
            tail = volumes.iloc[-lookback_days:] if len(volumes) > lookback_days else volumes
            avg_shares = float(tail.mean())

    if avg_shares is None or avg_shares <= 0:
        return None, None

    adv_cr = avg_shares * current_price_abs / 1e7
    return _round_cr(adv_cr), round(avg_shares, 0)
