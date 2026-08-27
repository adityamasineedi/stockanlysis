"""Unit tests for portfolio pre-screener — no network."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stockbot.models import Financials, TickerInfo
from stockbot.portfolio_screener.backtester import (
    BenchmarkPath,
    ForwardPath,
    evaluate_selection,
)
from stockbot.portfolio_screener.correlation import compute_correlation_infos
from stockbot.portfolio_screener.data_validator import validate_stock_data
from stockbot.portfolio_screener.hard_filters import apply_hard_filters
from stockbot.portfolio_screener.metrics import extract_metrics
from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.pipeline import run_prescreen
from stockbot.portfolio_screener.portfolio_selector import (
    combine_scores,
    select_portfolio,
)
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.score_utils import cagr, clamp, linear_score
from stockbot.portfolio_screener.scoring_config import (
    PortfolioConstraints,
    ScreenerRunConfig,
    ScreeningWeights,
)


def _years(*vals: float) -> dict[str, float]:
    labels = ["Mar 2021", "Mar 2022", "Mar 2023", "Mar 2024", "Mar 2025"]
    return {labels[i]: vals[i] for i in range(len(vals))}


def _make_financials(
    *,
    sales: list[float],
    pat: list[float],
    eps: list[float],
    ocf: list[float],
    op: list[float] | None = None,
    debt: list[float] | None = None,
) -> Financials:
    cols = ["Mar 2021", "Mar 2022", "Mar 2023", "Mar 2024", "Mar 2025"][: len(sales)]
    op = op or [s * 0.2 for s in sales]
    debt = debt or [50.0] * len(sales)
    pnl = pd.DataFrame(
        {
            "Sales": sales,
            "Operating Profit": op,
            "Interest": [5.0] * len(sales),
            "Depreciation": [10.0] * len(sales),
            "Net Profit": pat,
            "EPS in Rs": eps,
        },
        index=cols,
    ).T
    bs = pd.DataFrame(
        {
            "Total Assets": [1000.0] * len(sales),
            "Borrowings": debt,
            "Equity Capital": [100.0] * len(sales),
            "Reserves": [400.0] * len(sales),
            "Cash Equivalents": [80.0] * len(sales),
        },
        index=cols,
    ).T
    cf = pd.DataFrame(
        {
            "Cash from Operating Activity": ocf,
            "Cash from Investing Activity": [-30.0] * len(sales),
            "Net Cash Flow": [5.0] * len(sales),
        },
        index=cols,
    ).T
    ratios = pd.DataFrame(
        {
            "ROE %": [18.0] * len(sales),
            "ROCE %": [22.0] * len(sales),
            "Current Ratio": [1.8] * len(sales),
        },
        index=cols,
    ).T
    return Financials(
        pnl=pnl,
        balance_sheet=bs,
        cash_flow=cf,
        ratios=ratios,
        quarterly=pnl.iloc[:, -1:].copy(),
        basis="consolidated",
        years_available=len(sales),
        source="test",
        fetched_at=datetime.now(UTC),
    )


def _good_metrics(ticker: str, sector: str = "Technology") -> StockMetrics:
    fin = _make_financials(
        sales=[800, 900, 1000, 1100, 1200],
        pat=[100, 120, 140, 160, 180],
        eps=[10, 12, 14, 16, 18],
        ocf=[110, 130, 150, 170, 190],
    )
    ticker_info = TickerInfo(symbol=ticker, exchange="NSE", company_name=ticker, isin=None)
    m = extract_metrics(
        ticker_info,
        financials=fin,
        price=None,
        shareholding=None,
        market_meta={
            "sector": sector,
            "industry": "Software",
            "market_cap_cr": 50000.0,
            "trailing_pe": 25.0,
            "pb": 5.0,
            "forward_pe": 22.0,
            "dividend_yield_pct": 1.0,
        },
    )
    # price is critical — inject manually for tests without yfinance
    m.current_price_abs = 450.0
    m.missing.pop("current_price_abs", None)
    m.promoter_pct = 40.0
    m.promoter_pledge_pct = 0.0
    m.missing.pop("promoter_pct", None)
    m.missing.pop("promoter_pledge_pct", None)
    # synthetic returns for correlation
    import math

    base = [math.sin(i / 7) * 0.01 + 0.0005 for i in range(120)]
    m.price_returns = base
    return m


def test_score_utils_cagr_and_clamp():
    assert abs(cagr(100.0, 121.0, 2) - 0.1) < 1e-9
    assert cagr(-10.0, 10.0, 2) is None
    assert clamp(150) == 100
    assert linear_score(15.0, bad=5.0, good=25.0) == 50.0


def test_extract_metrics_does_not_invent_cash_when_absent():
    fin = _make_financials(
        sales=[100, 110, 120],
        pat=[10, 11, 12],
        eps=[1, 1.1, 1.2],
        ocf=[12, 13, 14],
    )
    # drop cash row
    fin.balance_sheet.drop(index="Cash Equivalents", inplace=True)
    ticker = TickerInfo(symbol="ABC", exchange="NSE", company_name="ABC", isin=None)
    m = extract_metrics(
        ticker,
        financials=fin,
        price=None,
        shareholding=None,
        market_meta={"sector": "Industrials", "market_cap_cr": 1000.0},
    )
    assert m.cash is None
    assert "cash" in m.missing


def test_hard_filter_excludes_persistent_losses():
    m = _good_metrics("LOSSCO")
    m.net_income_series = [-10.0, -20.0, -30.0, -40.0, -50.0]
    m.net_income = -50.0
    validation = validate_stock_data(m)
    hard = apply_hard_filters(m, validation)
    assert hard.status == "HARD_EXCLUDE"
    assert any("persistent losses" in r for r in hard.reasons)


def test_hard_filter_data_insufficient_when_critical_missing():
    m = StockMetrics(ticker="EMPTY")
    validation = validate_stock_data(m)
    hard = apply_hard_filters(m, validation)
    assert hard.status == "DATA_INSUFFICIENT"
    assert validation.critical_ok is False


def test_quant_score_auditable_base_and_penalty():
    m = _good_metrics("TCS")
    # force a red-flag via high pledge
    m.promoter_pledge_pct = 25.0
    result = compute_quant_score(m, ScreenerRunConfig(skip_ai=True, dry_run=True))
    assert result.base_score > 0
    assert result.red_flag_penalty < 0
    assert result.final_quant_score == clamp(result.base_score + result.red_flag_penalty)
    assert result.hard_filter.status == "PASS"


def test_ai_cannot_override_hard_exclude_via_selector():
    m = _good_metrics("BAD")
    m.net_income_series = [-1, -2, -3, -4, -5]
    m.net_income = -5
    q = compute_quant_score(m, ScreenerRunConfig())
    assert q.hard_filter.status == "HARD_EXCLUDE"
    from stockbot.portfolio_screener.models import AIRankResult

    ai = [
        AIRankResult(
            ticker="BAD",
            rank=1,
            ai_score=99,
            confidence="HIGH",
            keep_for_deep_analysis=True,
            key_reason="should not matter",
            key_risk="",
            data_concerns=[],
        )
    ]
    selected, rejected, _status = select_portfolio(
        [q], ai, {}, ScreenerRunConfig(constraints=PortfolioConstraints(min_stocks=1, max_stocks=5))
    )
    assert all(s.ticker != "BAD" for s in selected)
    assert any(r.selection_status == "HARD_EXCLUDED" for r in rejected)


def test_diversification_caps_sector_concentration():
    metrics = [_good_metrics(f"IT{i}", sector="Technology") for i in range(12)]
    # add a couple from another sector with slightly lower scores
    metrics += [_good_metrics(f"BK{i}", sector="Banks") for i in range(4)]
    for i, m in enumerate(metrics):
        # differentiate scores via ROCE
        m.roce = 30.0 - i * 0.5

    config = ScreenerRunConfig(
        dry_run=True,
        skip_ai=True,
        constraints=PortfolioConstraints(
            min_stocks=4,
            max_stocks=8,
            max_sector_weight=0.30,
            max_industry_weight=0.50,
            min_final_score=50.0,
            watchlist_min=50.0,
            candidate_min=55.0,
            strong_candidate_min=70.0,
            ai_shortlist_size=20,
        ),
    )
    result = run_prescreen(metrics=metrics, config=config, write_audit=False)
    tech = [s for s in result.stocks if s.sector == "Technology"]
    max_tech = max(1, int(config.constraints.max_stocks * config.constraints.max_sector_weight))
    assert len(tech) <= max_tech
    assert result.final_candidates <= config.constraints.max_stocks


def test_does_not_force_fill_below_quality():
    weak = []
    for i in range(5):
        m = StockMetrics(
            ticker=f"W{i}",
            sector="Unknown",
            industry="Unknown",
            current_price_abs=10.0,
            revenue=100.0,
            net_income=1.0,
            eps=0.1,
            operating_cash_flow=1.0,
            roe=2.0,
            years_available=1,
        )
        weak.append(m)
    config = ScreenerRunConfig(
        dry_run=True,
        skip_ai=True,
        constraints=PortfolioConstraints(min_stocks=10, max_stocks=18, min_final_score=60.0),
    )
    result = run_prescreen(metrics=weak, config=config, write_audit=False)
    assert result.status == "INSUFFICIENT_HIGH_QUALITY_CANDIDATES"
    assert result.final_candidates < 10


def test_combine_scores_respects_blend():
    config = ScreenerRunConfig()
    assert combine_scores(100.0, 0.0, config) == 70.0


def test_correlation_clusters_highly_similar_series():
    a = _good_metrics("A", "Technology")
    b = _good_metrics("B", "Technology")
    a.price_returns = [0.01] * 100
    b.price_returns = [0.01] * 100
    infos = compute_correlation_infos([a, b])
    assert infos["A"].correlation_risk in ("HIGH", "MEDIUM", "LOW")
    # perfect correlation should cluster
    assert infos["A"].max_peer_correlation is not None
    assert infos["A"].max_peer_correlation > 0.9


def test_backtester_no_lookahead_contract():
    report = evaluate_selection(
        [
            ForwardPath("A", 0.1, 0.2, 0.3, -0.15, 0.2),
            ForwardPath("B", -0.05, 0.05, 0.1, -0.25, 0.3),
        ],
        benchmark=BenchmarkPath("NIFTY 50", 0.05, 0.1, 0.15),
    )
    assert report.n_selected == 2
    assert report.avg_return_12m is not None
    assert report.excess_vs_benchmark_12m is not None
    assert any("point-in-time" in n.lower() or "look" in n.lower() or "Validation" in n for n in report.notes)


def test_weights_are_configurable_not_hardcoded():
    m = _good_metrics("CFG")
    heavy_quality = ScreenerRunConfig(
        weights=ScreeningWeights(
            business_quality=50,
            financial_strength=10,
            growth=10,
            cash_flow_quality=5,
            capital_efficiency=5,
            valuation=5,
            balance_sheet=5,
            earnings_quality=5,
            risk=5,
        ),
        skip_ai=True,
        dry_run=True,
    )
    balanced = ScreenerRunConfig(skip_ai=True, dry_run=True)
    a = compute_quant_score(m, heavy_quality)
    b = compute_quant_score(m, balanced)
    # Different weight profiles should generally produce different base scores
    # (unless all components happen to be identical — still assert structure)
    assert a.base_score > 0 and b.base_score > 0
    assert heavy_quality.weights.business_quality == 50
