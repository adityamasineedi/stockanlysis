"""Capital Efficiency (CE) pillar — ROIC + asset turnover only.

ROE / ROCE / OPM live in Quality (Q). Do not re-score them here — that
double-counted the same profitability stack and crushed mixed names.
"""

from __future__ import annotations

from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    finalize_pillar,
    linear_score,
    weighted_mean,
)


def score_capital_efficiency(metrics: StockMetrics) -> float:
    """Score how efficiently capital is deployed — not another ROE copy."""
    parts = [
        (linear_score(metrics.roic, bad=5.0, good=20.0), 0.60),
        (linear_score(metrics.asset_turnover, bad=0.3, good=1.5), 0.40),
    ]
    score, coverage = weighted_mean(parts)
    return finalize_pillar(score, coverage)
