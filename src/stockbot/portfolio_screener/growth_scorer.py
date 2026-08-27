"""Growth score with trend / quality labels."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    clamp,
    growth_trend_from_cagrs,
    linear_score,
    weighted_mean,
)
from stockbot.portfolio_screener.scoring_config import GrowthTrend


def score_growth(metrics: StockMetrics) -> tuple[float, float, GrowthTrend]:
    """Returns (growth_score, growth_quality, growth_trend).

    Extremely high growth is not blindly rewarded — quality discounts
    decelerating / volatile paths.
    """
    rev3 = metrics.revenue_cagr_3y
    rev5 = metrics.revenue_cagr_5y
    eps3 = metrics.eps_cagr_3y
    eps5 = metrics.eps_cagr_5y
    ebitda3 = metrics.ebitda_cagr_3y

    def _growth_piece(cagr: float | None) -> float | None:
        if cagr is None:
            return None
        # Sweet spot ~8–25%; above 40% gets mild discount unless stable.
        base = linear_score(cagr, bad=-0.05, good=0.20)
        if cagr > 0.40:
            base = min(base, 85.0)
        return base

    parts = [
        (_growth_piece(rev3), 0.30),
        (_growth_piece(rev5), 0.15),
        (_growth_piece(eps3), 0.25),
        (_growth_piece(eps5), 0.10),
        (_growth_piece(ebitda3), 0.20),
    ]
    score, coverage = weighted_mean(parts)
    trend_str = growth_trend_from_cagrs(rev3 if rev3 is not None else eps3, rev5 if rev5 is not None else eps5)
    trend: GrowthTrend = trend_str  # type: ignore[assignment]

    quality = score
    if trend == "ACCELERATING":
        quality = min(100.0, quality + 8.0)
    elif trend == "DECELERATING":
        quality = max(0.0, quality - 10.0)
    elif trend == "NEGATIVE":
        quality = max(0.0, quality - 25.0)

    adjusted = score * (0.6 + 0.4 * coverage)
    return clamp(adjusted), clamp(quality), trend
