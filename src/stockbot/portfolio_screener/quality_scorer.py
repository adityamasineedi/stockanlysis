"""Business quality score (0–100). Quantitative proxies only — no moat claims."""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    finalize_pillar,
    linear_score,
    stability_score,
    weighted_mean,
)
from stockbot.portfolio_screener.scoring_config import MoatConfidence


def score_business_quality(metrics: StockMetrics) -> tuple[float, MoatConfidence]:
    # Earnings-path stability lives in earnings_quality — do not re-score NI here.
    parts = [
        (linear_score(metrics.roce, bad=5.0, good=25.0), 0.25),
        (linear_score(metrics.roe, bad=5.0, good=22.0), 0.20),
        (linear_score(metrics.operating_margin, bad=0.05, good=0.25), 0.20),
        (linear_score(metrics.ebitda_margin, bad=0.08, good=0.30), 0.10),
        (stability_score(metrics.operating_margin_series), 0.125),
        (stability_score(metrics.revenue_series), 0.125),
    ]
    score, coverage = weighted_mean(parts)
    adjusted = finalize_pillar(score, coverage)
    moat: MoatConfidence = "LOW"
    if coverage >= 0.8 and score >= 75 and metrics.roce is not None and metrics.roce >= 20:
        moat = "MEDIUM"  # still not HIGH — qualitative moat needs more than ratios
    return adjusted, moat
