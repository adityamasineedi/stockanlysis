"""Integration-style tests for pre-screener pipeline (mocked fetch)."""

from __future__ import annotations

from unittest.mock import patch

from stockbot.models import TickerInfo
from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.pipeline import (
    run_prescreen,
    run_prescreen_then_analyze,
)
from stockbot.portfolio_screener.scoring_config import (
    PortfolioConstraints,
    ScreenerRunConfig,
)


def _metrics_batch(n: int = 15) -> list[StockMetrics]:
    out: list[StockMetrics] = []
    sectors = ["Technology", "Banks", "Healthcare", "Industrials", "Consumer Defensive"]
    for i in range(n):
        sector = sectors[i % len(sectors)]
        m = StockMetrics(
            ticker=f"S{i:02d}",
            company_name=f"Stock {i}",
            sector=sector,
            industry=f"{sector}-Ind",
            market_cap_cr=10000.0 + i * 100,
            current_price_abs=100.0 + i,
            revenue=1000.0 + i * 10,
            revenue_series=[800, 900, 1000, 1050, 1100 + i],
            net_income=100.0 + i,
            net_income_series=[60, 70, 80, 90, 100 + i],
            eps=10.0 + i * 0.1,
            eps_series=[6, 7, 8, 9, 10 + i * 0.1],
            operating_cash_flow=120.0 + i,
            ocf_series=[70, 80, 90, 100, 120 + i],
            free_cash_flow=80.0 + i,
            fcf_series=[40, 50, 60, 70, 80 + i],
            roe=15.0 + i * 0.2,
            roce=18.0 + i * 0.2,
            debt=50.0,
            cash=40.0,
            net_debt=10.0,
            equity=500.0,
            interest_coverage=8.0,
            debt_equity=0.1,
            operating_margin=0.18,
            ebitda_margin=0.22,
            pe=22.0 + i * 0.3,
            pb=3.0,
            ocf_to_pat=1.1,
            fcf_to_pat=0.8,
            fcf_margin=0.08,
            revenue_cagr_3y=0.10,
            eps_cagr_3y=0.12,
            promoter_pct=45.0,
            promoter_pledge_pct=0.0,
            years_available=5,
            price_returns=[0.001 * ((i % 5) - 2) for _ in range(80)],
        )
        out.append(m)
    return out


def test_end_to_end_prescreen_dry_run_produces_candidates():
    config = ScreenerRunConfig(
        dry_run=True,
        skip_ai=True,
        constraints=PortfolioConstraints(
            min_stocks=5,
            max_stocks=12,
            max_sector_weight=0.40,
            max_industry_weight=0.30,
            min_final_score=50.0,
            watchlist_min=50.0,
            candidate_min=55.0,
            strong_candidate_min=70.0,
        ),
    )
    result = run_prescreen(metrics=_metrics_batch(20), config=config, write_audit=False)
    assert result.universe_size == 20
    assert result.final_candidates >= 5
    assert result.final_candidates <= 12
    assert result.status in ("READY_FOR_DEEP_ANALYSIS", "DRY_RUN_COMPLETE")
    assert result.costs.stocks_processed == 20
    assert result.deep_analysis_tickers
    # Every stock accounted for
    all_tickers = {s.ticker for s in result.stocks} | {r.ticker for r in result.rejected}
    assert len(all_tickers) == 20
    payload = result.to_dict()
    assert "costs" in payload
    assert payload["screening_version"]


def test_handoff_not_invoked_on_dry_run():
    config = ScreenerRunConfig(
        dry_run=True,
        skip_ai=True,
        run_deep_analysis=True,
        constraints=PortfolioConstraints(
            min_stocks=3, max_stocks=8, min_final_score=50.0, watchlist_min=50.0
        ),
    )
    screened = run_prescreen(
        metrics=_metrics_batch(12),
        config=config,
        write_audit=False,
    )
    assert screened.status == "DRY_RUN_COMPLETE"

    with (
        patch(
            "stockbot.portfolio_screener.pipeline.run_prescreen",
            return_value=screened,
        ),
        patch("stockbot.pipeline.run_full_analysis") as mock_deep,
    ):
        out = run_prescreen_then_analyze(config=config)
        assert mock_deep.call_count == 0
        assert out.deep_analysis_results == []


def test_resolve_and_fetch_mocked_integration():
    tickers = [
        TickerInfo(symbol="AAA", exchange="NSE", company_name="AAA Ltd", isin=None),
        TickerInfo(symbol="BBB", exchange="NSE", company_name="BBB Ltd", isin=None),
    ]
    metrics = _metrics_batch(2)
    metrics[0].ticker = "AAA"
    metrics[1].ticker = "BBB"

    with (
        patch("stockbot.portfolio_screener.pipeline.load_watchlist", return_value=["AAA", "BBB"]),
        patch(
            "stockbot.portfolio_screener.pipeline.resolve_universe",
            return_value=type(
                "U",
                (),
                {
                    "tickers": tickers,
                    "unresolved": [],
                    "ambiguous": {},
                    "loaded_at": metrics[0].data_timestamp,
                },
            )(),
        ),
        patch(
            "stockbot.portfolio_screener.pipeline.fetch_universe_metrics",
            return_value=metrics,
        ),
    ):
        result = run_prescreen(
            config=ScreenerRunConfig(
                dry_run=True,
                skip_ai=True,
                constraints=PortfolioConstraints(min_stocks=1, max_stocks=5, min_final_score=40.0, watchlist_min=40.0),
            ),
            write_audit=False,
        )
    assert result.universe_size == 2
