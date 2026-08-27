"""Sector-aware valuation score."""

from __future__ import annotations

from stockbot.portfolio_screener.hard_filters import classify_valuation_risk
from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    clamp,
    linear_score,
    percentile_rank,
    weighted_mean,
)
from stockbot.portfolio_screener.scoring_config import (
    DEFAULT_SECTOR_BENCHMARKS,
    ConfidenceLevel,
    SectorValuationBenchmarks,
    ValuationRisk,
)


def _benchmark_for_sector(
    sector: str | None,
    benchmarks: dict[str, SectorValuationBenchmarks],
) -> SectorValuationBenchmarks:
    if sector and sector in benchmarks:
        return benchmarks[sector]
    # Fuzzy contains match
    if sector:
        lowered = sector.lower()
        for key, bench in benchmarks.items():
            if key.lower() in lowered or lowered in key.lower():
                return bench
    return benchmarks.get("Unknown", DEFAULT_SECTOR_BENCHMARKS["Unknown"])


def score_valuation(
    metrics: StockMetrics,
    *,
    peer_pes: list[float] | None = None,
    peer_evs: list[float] | None = None,
    benchmarks: dict[str, SectorValuationBenchmarks] | None = None,
) -> tuple[float, ValuationRisk, float | None, ConfidenceLevel]:
    """Returns (score, valuation_risk, sector_percentile, confidence).

    Historical valuation percentiles are never fabricated.
    """
    benches = benchmarks or DEFAULT_SECTOR_BENCHMARKS
    bench = _benchmark_for_sector(metrics.sector, benches)
    risk = classify_valuation_risk(metrics, sector_pe_expensive=bench.pe_expensive)

    pe_score = linear_score(
        metrics.pe,
        bad=bench.pe_expensive * 1.3,
        good=bench.pe_fair * 0.7,
        higher_is_better=False,
    )
    # Negative/zero PE already handled by linear_score None path if pe is None;
    # if pe is negative, treat as missing for valuation attractiveness.
    if metrics.pe is not None and metrics.pe <= 0:
        pe_score = 25.0

    ev_score = None
    if "ev_ebitda" in bench.preferred_metrics and bench.ev_ebitda_fair is not None:
        ev_score = linear_score(
            metrics.ev_ebitda,
            bad=bench.ev_ebitda_fair * 2.0,
            good=bench.ev_ebitda_fair * 0.7,
            higher_is_better=False,
        )

    pb_score = None
    if "pb" in bench.preferred_metrics and bench.pb_fair is not None:
        pb_score = linear_score(
            metrics.pb,
            bad=bench.pb_fair * 2.0,
            good=bench.pb_fair * 0.7,
            higher_is_better=False,
        )

    peg_score = linear_score(metrics.peg, bad=2.5, good=0.8, higher_is_better=False)
    pfcf_score = linear_score(
        metrics.price_fcf, bad=40.0, good=12.0, higher_is_better=False
    )

    sector_percentile = percentile_rank(metrics.pe, peer_pes or [])
    if sector_percentile is None and metrics.ev_ebitda is not None:
        sector_percentile = percentile_rank(metrics.ev_ebitda, peer_evs or [])

    # Lower PE percentile among peers = cheaper = higher valuation score contribution
    peer_score = None
    if sector_percentile is not None:
        peer_score = 100.0 - sector_percentile

    parts = [
        (pe_score, 0.35),
        (ev_score, 0.20),
        (pb_score, 0.15),
        (peg_score, 0.10),
        (pfcf_score, 0.10),
        (peer_score, 0.10),
    ]
    score, coverage = weighted_mean(parts)

    confidence: ConfidenceLevel
    if coverage >= 0.6 and sector_percentile is not None:
        confidence = "MEDIUM"
    elif coverage >= 0.4:
        confidence = "MEDIUM" if peer_score is not None else "LOW"
    else:
        confidence = "LOW"

    # High-quality growth can remain candidates despite expensive valuation —
    # we still score valuation honestly; selection logic keeps them if other
    # pillars compensate. No fabrication of historical percentiles.
    return clamp(score * (0.55 + 0.45 * coverage)), risk, sector_percentile, confidence
