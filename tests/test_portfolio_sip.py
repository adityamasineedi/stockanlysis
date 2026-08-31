"""Portfolio SIP allocation and messaging."""

from __future__ import annotations

from stockbot.portfolio_sip import (
    PortfolioBucket,
    PortfolioSipConfig,
    allocate_portfolio,
    build_portfolio_sip_plan,
    equal_whole_share_lines,
    load_portfolio_sip_config,
    priority_whole_share_lines,
    target_split_whole_share_lines,
)
from stockbot.portfolio_sip_messages import (
    format_portfolio_plan_html,
    format_portfolio_track_html,
    split_telegram_chunks,
)
from stockbot.portfolio_sip_schema import CashPolicy, SymbolConfig


def test_equal_whole_share_respects_budget():
    prices = {"A": 1000.0, "B": 1000.0, "C": 1000.0}
    lines = equal_whole_share_lines(("A", "B", "C"), 2500.0, prices)
    invested = sum(line.invested for line in lines)
    assert invested <= 2500.0
    assert sum(1 for line in lines if line.shares >= 1) >= 2


def test_target_split_respects_budget_and_targets():
    symbols = (
        SymbolConfig("A", target_amount_monthly=1000, max_amount_monthly=1500),
        SymbolConfig("B", target_amount_monthly=1000, max_amount_monthly=1500),
    )
    prices = {"A": 100.0, "B": 200.0}
    lines = target_split_whole_share_lines(symbols, 2000.0, prices, month=8)
    invested = sum(line.invested for line in lines)
    assert invested <= 2000.0
    by_sym = {line.symbol: line for line in lines}
    assert by_sym["A"].shares == 10
    assert by_sym["B"].shares == 5


def test_rotation_skips_netweb_in_august():
    symbols = (
        SymbolConfig(
            "NETWEB",
            target_amount_monthly=0,
            max_amount_monthly=5500,
            rotation=__import__(
                "stockbot.portfolio_sip_schema", fromlist=["RotationConfig"]
            ).RotationConfig(
                enabled=True,
                mode="round_robin",
                cycle_months=3,
                months_active=(1, 4, 7, 10),
            ),
        ),
    )
    prices = {"NETWEB": 5000.0}
    lines = target_split_whole_share_lines(symbols, 20000.0, prices, month=8)
    assert lines[0].rotation_skip
    assert lines[0].shares == 0


def test_rotation_active_netweb_in_january():
    from stockbot.portfolio_sip_schema import RotationConfig

    symbols = (
        SymbolConfig(
            "NETWEB",
            target_amount_monthly=0,
            max_amount_monthly=5500,
            rotation=RotationConfig(
                enabled=True,
                months_active=(1, 4, 7, 10),
            ),
        ),
    )
    prices = {"NETWEB": 5000.0}
    lines = target_split_whole_share_lines(symbols, 20000.0, prices, month=1)
    assert lines[0].shares == 1
    assert lines[0].invested == 5000.0


def test_overflow_symbol_adds_gold_shares():
    symbols = (
        SymbolConfig("GOLDBEES", target_amount_monthly=100, max_amount_monthly=4000),
        SymbolConfig("VEDL", target_amount_monthly=500, max_amount_monthly=5000),
    )
    prices = {"GOLDBEES": 100.0, "VEDL": 1000.0}
    lines = target_split_whole_share_lines(
        symbols,
        2000.0,
        prices,
        month=8,
        overflow_symbol="GOLDBEES",
    )
    by_sym = {line.symbol: line for line in lines}
    assert by_sym["VEDL"].shares == 0
    assert by_sym["GOLDBEES"].shares == 20


def test_priority_fills_p1_before_expensive_p3():
    prices = {"MAZDOCK": 2480.0, "MOSCHIP": 214.0, "NETWEB": 5188.0}
    lines = priority_whole_share_lines(("MAZDOCK", "MOSCHIP", "NETWEB"), 20000.0, prices)
    by_sym = {line.symbol: line for line in lines}
    assert by_sym["MAZDOCK"].priority_rank == 1
    assert by_sym["MAZDOCK"].shares >= 1
    assert by_sym["MOSCHIP"].shares >= 1
    assert sum(line.invested for line in lines) <= 20000.0


def test_allocate_portfolio_computes_cash_aside():
    portfolio = PortfolioBucket(
        "x",
        "Test",
        2000.0,
        (SymbolConfig("A"), SymbolConfig("B")),
        allocation_mode="equal",
    )
    prices = {"A": 100.0, "B": 200.0}
    allocation = allocate_portfolio(portfolio, prices, month=8)
    assert allocation.invested <= 2000.0
    assert allocation.cash_aside == round(2000.0 - allocation.invested, 2)


def test_build_portfolio_sip_plan_with_stub_fetcher():
    config = PortfolioSipConfig(
        version=1,
        currency="INR",
        max_monthly_budget=4000.0,
        whole_shares_only=True,
        broker_rounding="floor",
        portfolios=(
            PortfolioBucket(
                "a",
                "Bucket A",
                2000.0,
                (SymbolConfig("AAA"), SymbolConfig("BBB")),
                allocation_mode="equal_split",
            ),
            PortfolioBucket("b", "Bucket B", 2000.0, (SymbolConfig("CCC"),)),
        ),
    )

    def fetcher(symbol: str) -> tuple[float | None, float | None, float | None]:
        return (100.0, 110.0, 120.0)

    plan = build_portfolio_sip_plan(config, fetcher=fetcher)
    assert plan.total_invested <= 4000.0
    assert len(plan.allocations) == 2


def test_format_plan_and_track_html():
    config = PortfolioSipConfig(
        version=1,
        currency="INR",
        max_monthly_budget=2000.0,
        whole_shares_only=True,
        broker_rounding="floor",
        portfolios=(
            PortfolioBucket(
                "g",
                "Growth",
                2000.0,
                (
                    SymbolConfig("BSE", target_amount_monthly=1000),
                    SymbolConfig("MCX", target_amount_monthly=1000),
                ),
                allocation_mode="equal_split",
                thesis="Financial infrastructure — moves together on policy news.",
            ),
        ),
        default_allocation_mode="equal_split",
    )
    plan = build_portfolio_sip_plan(
        config,
        fetcher=lambda _s: (1000.0, 1100.0, 1200.0),
    )
    html = format_portfolio_plan_html(plan)
    assert "Portfolio SIP plan" in html
    assert "BSE" in html
    assert "Growth" in html
    assert "target" in html.lower()

    track = format_portfolio_track_html(plan, {"BSE": 1000.0}, month_label="August 2026")
    assert "SIP track" in track
    assert "✅" in track or "⏳" in track


def test_split_telegram_chunks_on_blank_lines():
    text = "a\n\n" + ("x" * 3000) + "\n\n" + ("y" * 2000)
    chunks = split_telegram_chunks(text, limit=4000)
    assert len(chunks) >= 2
    assert all(len(c) <= 4000 for c in chunks)


def test_load_portfolio_sip_config_v1_from_repo_file():
    cfg = load_portfolio_sip_config()
    assert cfg.version == 1
    assert cfg.max_monthly_budget == 60000
    assert cfg.default_allocation_mode == "equal_split"
    assert len(cfg.portfolios) == 3
    bucket1 = cfg.portfolios[0]
    assert bucket1.symbols[0].symbol == "MAZDOCK"
    assert bucket1.thesis and "sector rotations" in bucket1.thesis
    netweb = next(s for s in bucket1.symbols if s.symbol == "NETWEB")
    assert netweb.rotation.enabled
    assert netweb.rotation.months_active == (1, 4, 7, 10)
    metals = cfg.portfolios[2]
    metals = cfg.portfolios[2]
    assert metals.cash_policy.overflow_symbol == "GOLDBEES"
    assert cfg.prescan_gate.enabled
    assert cfg.prescan_gate.monthly_auto_prescan
