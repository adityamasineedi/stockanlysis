"""Capital efficiency score (0–100)."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import clamp, linear_score, weighted_mean


def score_capital_efficiency(metrics: StockMetrics) -> float:
    parts = [
        (linear_score(metrics.roe, bad=5.0, good=22.0), 0.30),
        (linear_score(metrics.roce, bad=5.0, good=25.0), 0.35),
        (linear_score(metrics.roic, bad=5.0, good=20.0), 0.10),
        (linear_score(metrics.asset_turnover, bad=0.3, good=1.5), 0.15),
        (linear_score(metrics.operating_margin, bad=0.05, good=0.25), 0.10),
    ]
    score, coverage = weighted_mean(parts)
    return clamp(score * (0.6 + 0.4 * coverage))
