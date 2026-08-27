"""Cash-flow quality score (0–100)."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    clamp,
    linear_score,
    series_present,
    stability_score,
    weighted_mean,
)


def score_cash_flow(metrics: StockMetrics) -> float:
    ocf_consistency: float | None = None
    ocf = series_present(metrics.ocf_series)
    if ocf:
        positive_ratio = sum(1 for v in ocf if v > 0) / len(ocf)
        ocf_consistency = positive_ratio * 100.0

    fcf_consistency: float | None = None
    fcf = series_present(metrics.fcf_series)
    if fcf:
        positive_ratio = sum(1 for v in fcf if v > 0) / len(fcf)
        fcf_consistency = positive_ratio * 100.0

    # Working-capital behaviour proxied by OCF stability when debtor days absent.
    parts = [
        (linear_score(metrics.ocf_to_pat, bad=0.3, good=1.2), 0.30),
        (linear_score(metrics.fcf_to_pat, bad=0.0, good=1.0), 0.20),
        (linear_score(metrics.fcf_margin, bad=-0.05, good=0.15), 0.15),
        (ocf_consistency, 0.15),
        (fcf_consistency, 0.10),
        (stability_score(metrics.ocf_series, prefer_positive=True), 0.10),
    ]
    score, coverage = weighted_mean(parts)
    return clamp(score * (0.6 + 0.4 * coverage))
