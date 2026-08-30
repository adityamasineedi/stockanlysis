"""Aggregate quantitative scoring for one stock."""

from __future__ import annotations

from stockbot.portfolio_screener.capital_efficiency import score_capital_efficiency
from stockbot.portfolio_screener.cashflow_scorer import score_cash_flow
from stockbot.portfolio_screener.data_validator import validate_stock_data
from stockbot.portfolio_screener.financial_strength import score_financial_strength
from stockbot.portfolio_screener.growth_scorer import score_growth
from stockbot.portfolio_screener.hard_filters import apply_hard_filters
from stockbot.portfolio_screener.models import (
    ComponentScores,
    QuantScreenResult,
    StockMetrics,
)
from stockbot.portfolio_screener.quality_scorer import score_business_quality
from stockbot.portfolio_screener.red_flags import collect_red_flags, total_penalty
from stockbot.portfolio_screener.risk_scorer import (
    score_balance_sheet,
    score_earnings_quality,
    score_risk,
)
from stockbot.portfolio_screener.score_utils import clamp
from stockbot.portfolio_screener.scoring_config import (
    ScreenerRunConfig,
    SectorValuationBenchmarks,
)
from stockbot.portfolio_screener.sector_normalizer import is_cyclical_sector
from stockbot.portfolio_screener.valuation_scorer import score_valuation


def compute_quant_score(
    metrics: StockMetrics,
    config: ScreenerRunConfig,
    *,
    peer_pes: list[float] | None = None,
    peer_evs: list[float] | None = None,
    human_override: bool = False,
) -> QuantScreenResult:
    validation = validate_stock_data(metrics, config.hard_filters)

    sector_key = metrics.sector or "Unknown"
    bench: SectorValuationBenchmarks | None = config.sector_benchmarks.get(sector_key)
    expensive = bench.pe_expensive if bench else None

    hard = apply_hard_filters(
        metrics,
        validation,
        config.hard_filters,
        sector_pe_expensive=expensive,
        human_override=human_override,
    )

    bq, moat = score_business_quality(metrics)
    fs = score_financial_strength(metrics)
    growth, growth_quality, growth_trend = score_growth(metrics)
    cf = score_cash_flow(metrics)
    ce = score_capital_efficiency(metrics)
    val, val_risk, val_pct, val_conf = score_valuation(
        metrics,
        peer_pes=peer_pes,
        peer_evs=peer_evs,
        benchmarks=config.sector_benchmarks,
    )
    # Prefer hard-filter valuation risk classification when set
    if hard.valuation_risk is not None:
        val_risk = hard.valuation_risk

    bs = score_balance_sheet(metrics)
    eq = score_earnings_quality(metrics)
    risk = score_risk(
        metrics,
        valuation_risk=val_risk,
        cyclical_sector=is_cyclical_sector(metrics.sector, metrics.industry),
    )

    components = ComponentScores(
        business_quality=bq,
        financial_strength=fs,
        growth=growth,
        cash_flow_quality=cf,
        capital_efficiency=ce,
        valuation=val,
        balance_sheet=bs,
        earnings_quality=eq,
        risk=risk,
        growth_quality=growth_quality,
        growth_trend=growth_trend,
        valuation_risk=val_risk,
        valuation_percentile=val_pct,
        valuation_confidence=val_conf,
        moat_confidence=moat,
    )

    w = config.weights
    # Component scores are 0–100; weights sum to ~100 points.
    base = (
        bq * w.business_quality
        + fs * w.financial_strength
        + growth * w.growth
        + cf * w.cash_flow_quality
        + ce * w.capital_efficiency
        + val * w.valuation
        + bs * w.balance_sheet
        + eq * w.earnings_quality
        + risk * w.risk
    ) / w.total()

    # Soft discount for low data confidence — does not invent metrics.
    if validation.data_confidence == "LOW":
        base *= 0.90
    elif validation.data_confidence == "MEDIUM":
        base *= 0.96

    flags = collect_red_flags(metrics, config.red_flag_penalties)
    penalty = total_penalty(flags)
    final = clamp(base + penalty)

    return QuantScreenResult(
        ticker=metrics.ticker,
        base_score=round(base, 2),
        red_flag_penalty=round(penalty, 2),
        final_quant_score=round(final, 2),
        components=components,
        red_flags=flags,
        data_validation=validation,
        hard_filter=hard,
        sector=metrics.sector,
        industry=metrics.industry,
        data_timestamp=metrics.data_timestamp,
        current_price_abs=metrics.current_price_abs,
    )
