"""Sector / industry peer normalization."""

from __future__ import annotations

from collections import defaultdict

from stockbot.portfolio_screener.models import QuantScreenResult, StockMetrics
from stockbot.portfolio_screener.score_utils import percentile_rank

_CYCLICAL_SECTOR_KEYWORDS = (
    "basic materials",
    "energy",
    "industrials",
    "consumer cyclical",
    "real estate",
    "metals",
    "mining",
    "automobile",
    "auto",
)


def is_cyclical_sector(sector: str | None, industry: str | None = None) -> bool:
    blob = f"{sector or ''} {industry or ''}".lower()
    return any(k in blob for k in _CYCLICAL_SECTOR_KEYWORDS)


def peer_metric_map(
    universe: list[StockMetrics],
    attr: str,
) -> dict[str, list[float]]:
    """Map sector → list of peer values for `attr`."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for m in universe:
        sector = m.sector or "Unknown"
        value = getattr(m, attr, None)
        if isinstance(value, (int, float)):
            buckets[sector].append(float(value))
    return dict(buckets)


def attach_peer_percentiles(
    metrics: StockMetrics,
    *,
    sector_peers: dict[str, list[float]],
    industry_peers: dict[str, list[float]],
    metric_attr: str = "roce",
) -> tuple[float | None, float | None]:
    sector = metrics.sector or "Unknown"
    industry = metrics.industry or "Unknown"
    value = getattr(metrics, metric_attr, None)
    if not isinstance(value, (int, float)):
        return None, None
    sector_pct = percentile_rank(float(value), sector_peers.get(sector, []))
    industry_pct = percentile_rank(float(value), industry_peers.get(industry, []))
    return sector_pct, industry_pct


def sector_specific_notes(metrics: StockMetrics) -> list[str]:
    """Document which sector lens applies — scoring modules remain general
    but callers can log the intended emphasis."""
    sector = (metrics.sector or "").lower()
    notes: list[str] = []
    if any(k in sector for k in ("bank", "financial")):
        notes.append("Bank/Financial lens: prefer P/B, ROE; EV/EBITDA less relevant")
    elif any(k in sector for k in ("technology", "information technology")):
        notes.append("IT lens: prefer revenue growth, EBIT/FCF margins, ROCE, P/E")
    elif any(k in sector for k in ("industrial", "manufacturing", "basic materials")):
        notes.append("Manufacturing lens: prefer ROCE, EBITDA margin, FCF, debt")
    else:
        notes.append("Default multi-factor lens")
    return notes


def build_peer_pe_lists(universe: list[StockMetrics]) -> dict[str, list[float]]:
    return peer_metric_map(universe, "pe")


def build_peer_ev_lists(universe: list[StockMetrics]) -> dict[str, list[float]]:
    return peer_metric_map(universe, "ev_ebitda")


def industry_peer_map(universe: list[StockMetrics], attr: str) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for m in universe:
        industry = m.industry or "Unknown"
        value = getattr(m, attr, None)
        if isinstance(value, (int, float)):
            buckets[industry].append(float(value))
    return dict(buckets)


def enrich_quant_with_sector_percentiles(
    result: QuantScreenResult,
    metrics: StockMetrics,
    universe: list[StockMetrics],
) -> QuantScreenResult:
    sector_roce = peer_metric_map(universe, "roce")
    industry_roce = industry_peer_map(universe, "roce")
    s_pct, i_pct = attach_peer_percentiles(
        metrics, sector_peers=sector_roce, industry_peers=industry_roce
    )
    result.components.sector_percentile = s_pct
    result.components.industry_percentile = i_pct
    return result
