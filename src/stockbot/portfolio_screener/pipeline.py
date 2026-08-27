"""Portfolio pre-screener orchestration.

Universe → validate → hard filter → quant score → AI rank → diversify
→ 10–18 candidates → optional deep-analysis handoff.

Does NOT produce BUY/WATCH/SKIP / fair value / targets.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from stockbot.portfolio_screener.ai_ranker import rank_with_ai
from stockbot.portfolio_screener.audit_logger import (
    format_human_table,
    log_stock_decision,
    write_audit_artifact,
)
from stockbot.portfolio_screener.correlation import compute_correlation_infos
from stockbot.portfolio_screener.cost_tracker import ScreenerCostTracker
from stockbot.portfolio_screener.data_loader import (
    fetch_universe_metrics,
    load_watchlist,
    resolve_universe,
)
from stockbot.portfolio_screener.models import ScreeningResult, StockMetrics
from stockbot.portfolio_screener.portfolio_selector import select_portfolio
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig
from stockbot.portfolio_screener.sector_normalizer import (
    build_peer_ev_lists,
    build_peer_pe_lists,
    enrich_quant_with_sector_percentiles,
)

logger = logging.getLogger(__name__)


def run_prescreen(
    symbols: list[str] | None = None,
    *,
    watchlist_path: Path | None = None,
    config: ScreenerRunConfig | None = None,
    metrics: list[StockMetrics] | None = None,
    write_audit: bool = True,
) -> ScreeningResult:
    """Run the full pre-screener.

    If `metrics` is provided, skips network fetch (for tests / replay).
    If `symbols` is None, loads from watchlist_path / default watchlist.
    """
    config = config or ScreenerRunConfig()
    config.blend.validate()
    if abs(config.weights.total() - 100.0) > 1e-6:
        logger.warning(
            "SCREENING_WEIGHTS total is %s (expected 100) — normalising via total()",
            config.weights.total(),
        )

    universe_ts = datetime.now(UTC)
    if metrics is None:
        if symbols is None:
            symbols = load_watchlist(watchlist_path)
        loaded = resolve_universe(symbols)
        logger.info(
            "Resolved universe size=%d unresolved=%d ambiguous=%d",
            len(loaded.tickers),
            len(loaded.unresolved),
            len(loaded.ambiguous),
        )
        universe_ts = loaded.loaded_at
        metrics = fetch_universe_metrics(loaded.tickers)
    else:
        symbols = [m.ticker for m in metrics]

    tracker = ScreenerCostTracker(universe_size=len(metrics))
    peer_pe = build_peer_pe_lists(metrics)
    peer_ev = build_peer_ev_lists(metrics)

    quant_results = []
    for m in metrics:
        tracker.record_stock_processed()
        sector = m.sector or "Unknown"
        q = compute_quant_score(
            m,
            config,
            peer_pes=peer_pe.get(sector, []),
            peer_evs=peer_ev.get(sector, []),
        )
        q = enrich_quant_with_sector_percentiles(q, m, metrics)
        quant_results.append(q)
        logger.info(
            "quant ticker=%s status=%s score=%.1f base=%.1f penalty=%.1f",
            q.ticker,
            q.hard_filter.status,
            q.final_quant_score,
            q.base_score,
            q.red_flag_penalty,
        )

    passed = [q for q in quant_results if q.hard_filter.status == "PASS"]
    hard_excluded = sum(1 for q in quant_results if q.hard_filter.status == "HARD_EXCLUDE")
    data_insufficient = sum(
        1 for q in quant_results if q.hard_filter.status == "DATA_INSUFFICIENT"
    )

    ai_results = rank_with_ai(quant_results, config, cost_tracker=tracker)
    corr_map = compute_correlation_infos(metrics, config.constraints)

    selected, rejected, status = select_portfolio(
        quant_results, ai_results, corr_map, config
    )

    if config.dry_run and status == "READY_FOR_DEEP_ANALYSIS":
        status = "DRY_RUN_COMPLETE"

    all_for_table = sorted(
        selected,
        key=lambda r: r.ranking or 999,
    )
    human_table = format_human_table(all_for_table + [
        r for r in rejected if r.hard_filter_status != "PASS"
    ][:20])

    for rec in selected + rejected:
        log_stock_decision(rec)

    costs = tracker.summary(final_candidates=len(selected))
    result = ScreeningResult(
        universe_size=len(metrics),
        hard_excluded=hard_excluded,
        data_insufficient=data_insufficient,
        quant_screened=len(passed),
        sent_to_ai=len(ai_results),
        final_candidates=len(selected),
        status=status,  # type: ignore[arg-type]
        stocks=selected,
        rejected=rejected,
        costs=costs,
        screening_version=config.screening_version,
        weights_version=config.weights_version,
        prompt_version=config.prompt_version,
        ai_model=config.ai_model if not (config.skip_ai or config.dry_run) else "deterministic_fallback",
        data_timestamp=datetime.now(UTC),
        universe_timestamp=universe_ts,
        human_table=human_table,
        deep_analysis_tickers=[s.ticker for s in selected],
    )

    if write_audit:
        write_audit_artifact(result)

    logger.info(
        "prescreen done status=%s universe=%d excluded=%d candidates=%d "
        "ai_calls=%d cost_inr=%.2f estimated_savings_inr=%.2f",
        result.status,
        result.universe_size,
        result.hard_excluded,
        result.final_candidates,
        result.costs.ai_calls,
        result.costs.estimated_cost_inr,
        result.costs.estimated_deep_analysis_cost_saved_inr,
    )
    return result


def handoff_to_deep_analysis(
    result: ScreeningResult,
    *,
    max_analyses: int | None = None,
) -> ScreeningResult:
    """Run existing run_full_analysis on selected tickers only.

    Imported lazily to avoid circular imports and to keep dry-run free of
    LLM pipeline dependencies when unused.
    """
    from stockbot.pipeline import run_full_analysis

    tickers = result.deep_analysis_tickers
    if max_analyses is not None:
        tickers = tickers[:max_analyses]

    deep_results = []
    for symbol in tickers:
        logger.info("Deep analysis handoff: %s", symbol)
        deep_results.append(run_full_analysis(symbol))

    result.deep_analysis_results = deep_results
    return result


def run_prescreen_then_analyze(
    symbols: list[str] | None = None,
    *,
    watchlist_path: Path | None = None,
    config: ScreenerRunConfig | None = None,
) -> ScreeningResult:
    config = config or ScreenerRunConfig()
    result = run_prescreen(symbols, watchlist_path=watchlist_path, config=config)
    if config.dry_run or not config.run_deep_analysis:
        return result
    if result.status != "READY_FOR_DEEP_ANALYSIS":
        logger.warning(
            "Skipping deep analysis handoff — status=%s candidates=%d",
            result.status,
            result.final_candidates,
        )
        return result
    return handoff_to_deep_analysis(result, max_analyses=config.max_deep_analyses)
