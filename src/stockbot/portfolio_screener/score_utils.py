"""Pure numeric helpers shared by scorers. No I/O."""

from __future__ import annotations

import math
from statistics import pstdev


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def cagr(first: float | None, last: float | None, years: float) -> float | None:
    """Compound annual growth rate. Returns None if inputs invalid."""
    if first is None or last is None or years <= 0:
        return None
    if first <= 0 or last <= 0:
        return None
    return (last / first) ** (1.0 / years) - 1.0


def series_present(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def latest(values: list[float | None]) -> float | None:
    present = series_present(values)
    return present[-1] if present else None


def stability_score(values: list[float | None], *, prefer_positive: bool = True) -> float | None:
    """Higher = more stable. Based on coefficient of variation."""
    present = series_present(values)
    if len(present) < 2:
        return None
    mean = sum(present) / len(present)
    if mean == 0:
        return 40.0 if not prefer_positive else None
    if prefer_positive and mean < 0:
        return 20.0
    cv = pstdev(present) / abs(mean)
    # cv 0 → 100, cv 0.5 → ~50, cv >= 1 → ~10
    return clamp(100.0 - cv * 100.0)


def linear_score(
    value: float | None,
    *,
    bad: float,
    good: float,
    higher_is_better: bool = True,
) -> float | None:
    """Map a metric onto 0–100 between bad and good anchors."""
    if value is None:
        return None
    if higher_is_better:
        if value <= bad:
            return 0.0
        if value >= good:
            return 100.0
        return ((value - bad) / (good - bad)) * 100.0
    if value >= bad:
        return 0.0
    if value <= good:
        return 100.0
    return ((bad - value) / (bad - good)) * 100.0


def weighted_mean(scores: list[tuple[float | None, float]]) -> tuple[float, float]:
    """Return (weighted_average, coverage_fraction) ignoring None scores.

    coverage_fraction is sum(weights of available) / sum(all weights).
    When nothing is available, returns (0.0, 0.0).
    """
    total_w = sum(w for _, w in scores)
    if total_w <= 0:
        return 0.0, 0.0
    avail = [(s, w) for s, w in scores if s is not None]
    if not avail:
        return 0.0, 0.0
    avail_w = sum(w for _, w in avail)
    avg = sum(s * w for s, w in avail) / avail_w
    return avg, avail_w / total_w


def growth_trend_from_cagrs(
    recent: float | None,
    longer: float | None,
) -> str:
    if recent is None and longer is None:
        return "STABLE"
    if recent is not None and recent < 0:
        return "NEGATIVE"
    if recent is None:
        if longer is not None and longer < 0:
            return "NEGATIVE"
        return "STABLE"
    if longer is None:
        return "STABLE" if recent >= 0 else "NEGATIVE"
    delta = recent - longer
    if delta >= 0.03:
        return "ACCELERATING"
    if delta <= -0.03:
        return "DECELERATING"
    return "STABLE"


def percentile_rank(value: float | None, peers: list[float]) -> float | None:
    """Percentile of value within peers (0–100). None if insufficient."""
    if value is None or len(peers) < 2:
        return None
    below = sum(1 for p in peers if p < value)
    return (below / (len(peers) - 1)) * 100.0 if len(peers) > 1 else 50.0
