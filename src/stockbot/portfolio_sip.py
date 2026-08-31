"""Multi-stock portfolio SIP — whole-share allocation per bucket.

Reads bucket definitions from ``data/portfolio/sip_portfolios.json`` (v1 schema).
Pure allocation maths lives here; Telegram wording is in ``portfolio_sip_messages``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from stockbot.fetch.prices import fetch_price_data
from stockbot.portfolio_sip_prescan import (
    evaluate_prescan_gate,
    prescan_outcome_map,
    rank_symbols_by_prescan,
)
from stockbot.portfolio_sip_schema import (
    PortfolioBucket,
    PortfolioSipConfig,
    PrescanGateConfig,
    SymbolConfig,
    is_rotation_active,
    load_portfolio_sip_config,
    symbol_names,
)
from stockbot.sip import (
    classify_dip,
    dip_pct_from_high,
    suggest_topup,
    three_month_high,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Back-compat alias for message formatters.
PortfolioConfig = PortfolioBucket

PriceFetcher = Callable[[str], tuple[float | None, float | None, float | None]]


@dataclass(frozen=True)
class AllocationLine:
    symbol: str
    price: float | None
    shares: int
    invested: float
    priority_rank: int | None = None
    error: str | None = None
    dip_pct: float | None = None
    dip_label: str | None = None
    topup_range: tuple[float, float] | None = None
    rotation_skip: bool = False
    prescan_skip: bool = False
    note: str | None = None


@dataclass(frozen=True)
class PortfolioAllocation:
    portfolio: PortfolioBucket
    lines: tuple[AllocationLine, ...]
    invested: float
    cash_aside: float


@dataclass(frozen=True)
class PortfolioSipPlan:
    config: PortfolioSipConfig
    allocations: tuple[PortfolioAllocation, ...]
    total_invested: float
    total_cash_aside: float


def _floor_shares(amount: float, price: float) -> int:
    if price <= 0 or amount <= 0:
        return 0
    return int(amount // price)


def _clamp_target(amount: float, symbol: SymbolConfig) -> float:
    low = max(symbol.min_amount_monthly, 0.0)
    high = symbol.max_amount_monthly if symbol.max_amount_monthly > 0 else amount
    return max(low, min(amount, high))


def _monthly_target(symbol: SymbolConfig, month: int) -> float | None:
    if not symbol.enabled:
        return None
    if not is_rotation_active(symbol, month):
        return None
    if symbol.rotation.enabled and symbol.target_amount_monthly <= 0:
        return symbol.max_amount_monthly if symbol.max_amount_monthly > 0 else None
    if symbol.target_amount_monthly <= 0:
        return None
    return _clamp_target(symbol.target_amount_monthly, symbol)


def evaluate_symbol_dip(
    current_price: float,
    symbol: SymbolConfig,
    *,
    avg_cost: float | None,
    high_3m: float | None,
    week52_high: float | None,
) -> tuple[float | None, tuple[float, float] | None]:
    dip = symbol.dip
    if not dip.enabled or dip.trigger_type == "none":
        return None, None

    reference: float | None = None
    ref_key = (dip.reference or "").lower()
    if ref_key == "avg_cost":
        reference = avg_cost
    elif ref_key in {"52_week_high", "52w_high"}:
        reference = week52_high or high_3m
    else:
        reference = high_3m or week52_high

    if reference is None or reference <= 0 or current_price <= 0:
        return None, None

    drawdown = round((1.0 - current_price / reference) * 100.0, 2)
    if drawdown < dip.threshold_pct:
        return drawdown, None

    low = max(dip.band_min, symbol.topup.min_amount)
    high = min(dip.band_max, symbol.topup.max_amount) if symbol.topup.max_amount > 0 else dip.band_max
    if high <= 0 or low <= 0:
        return drawdown, None
    if low > high:
        low = dip.band_min
        high = dip.band_max
    return drawdown, (round(low, 2), round(high, 2))


def _build_line(
    symbol: SymbolConfig,
    shares: int,
    price: float | None,
    *,
    priority_rank: int | None = None,
    error: str | None = None,
    rotation_skip: bool = False,
    prescan_skip: bool = False,
    note: str | None = None,
    avg_cost: float | None = None,
    high_3m: float | None = None,
    week52_high: float | None = None,
) -> AllocationLine:
    if error or price is None or price <= 0:
        return AllocationLine(
            symbol=symbol.symbol,
            price=price,
            shares=0,
            invested=0.0,
            priority_rank=priority_rank,
            error=error or ("price unavailable" if price is None else None),
            rotation_skip=rotation_skip,
            prescan_skip=prescan_skip,
            note=note,
        )
    invested = round(shares * price, 2)
    dip_pct, topup = evaluate_symbol_dip(
        price,
        symbol,
        avg_cost=avg_cost,
        high_3m=high_3m,
        week52_high=week52_high,
    )
    dip_label = "DIP" if topup else None
    return AllocationLine(
        symbol=symbol.symbol,
        price=price,
        shares=shares,
        invested=invested,
        priority_rank=priority_rank,
        dip_pct=dip_pct,
        dip_label=dip_label,
        topup_range=topup,
        rotation_skip=rotation_skip,
        prescan_skip=prescan_skip,
        note=note,
    )


def target_split_whole_share_lines(
    symbols: tuple[SymbolConfig, ...],
    budget: float,
    prices: dict[str, float | None],
    month: int,
    *,
    errors: dict[str, str] | None = None,
    highs_3m: dict[str, float | None] | None = None,
    week52_highs: dict[str, float | None] | None = None,
    avg_costs: dict[str, float | None] | None = None,
    overflow_symbol: str | None = None,
    prescan_gate: PrescanGateConfig | None = None,
    prescan_map: dict[str, dict] | None = None,
) -> tuple[AllocationLine, ...]:
    """Allocate whole shares from per-symbol monthly targets within *budget*."""
    err = errors or {}
    highs = highs_3m or {}
    w52 = week52_highs or {}
    costs = avg_costs or {}
    gate = prescan_gate or PrescanGateConfig(enabled=False)
    pmap = prescan_map or {}

    rows: list[tuple[SymbolConfig, float, int, float]] = []
    skip_lines: list[AllocationLine] = []

    for index, symbol in enumerate(symbols):
        sym = symbol.symbol
        rank = index + 1
        if not symbol.enabled:
            continue
        if sym in err:
            skip_lines.append(
                _build_line(symbol, 0, prices.get(sym), priority_rank=rank, error=err[sym])
            )
            continue
        px = prices.get(sym)
        if px is None or px <= 0:
            skip_lines.append(
                _build_line(symbol, 0, px, priority_rank=rank, error="price unavailable")
            )
            continue
        if not is_rotation_active(symbol, month):
            skip_lines.append(
                _build_line(
                    symbol,
                    0,
                    px,
                    priority_rank=rank,
                    rotation_skip=True,
                    note="rotation skip",
                    avg_cost=costs.get(sym),
                    high_3m=highs.get(sym),
                    week52_high=w52.get(sym),
                )
            )
            continue
        gate_result = evaluate_prescan_gate(sym, pmap.get(sym), gate)
        if gate_result.blocked:
            skip_lines.append(
                _build_line(
                    symbol,
                    0,
                    px,
                    priority_rank=rank,
                    prescan_skip=True,
                    note=gate_result.note,
                    avg_cost=costs.get(sym),
                    high_3m=highs.get(sym),
                    week52_high=w52.get(sym),
                )
            )
            continue
        target = _monthly_target(symbol, month)
        if target is None:
            skip_lines.append(
                _build_line(
                    symbol,
                    0,
                    px,
                    priority_rank=rank,
                    note="no target",
                    avg_cost=costs.get(sym),
                    high_3m=highs.get(sym),
                    week52_high=w52.get(sym),
                )
            )
            continue
        shares = _floor_shares(target, px)
        invested = round(shares * px, 2)
        max_cap = symbol.max_amount_monthly if symbol.max_amount_monthly > 0 else invested
        while shares > 0 and invested > max_cap:
            shares -= 1
            invested = round(shares * px, 2)
        rows.append((symbol, px, shares, invested))

    total = sum(row[3] for row in rows)
    while total > budget and rows:
        idx = max(range(len(rows)), key=lambda i: rows[i][3])
        sym_cfg, px, shares, invested = rows[idx]
        if shares <= 0:
            rows.pop(idx)
            total = sum(row[3] for row in rows)
            continue
        shares -= 1
        invested = round(shares * px, 2)
        rows[idx] = (sym_cfg, px, shares, invested)
        total = sum(row[3] for row in rows)

    remainder = round(budget - total, 2)
    if remainder > 0 and overflow_symbol:
        for i, (sym_cfg, px, _, _) in enumerate(rows):
            if sym_cfg.symbol != overflow_symbol:
                continue
            max_cap = (
                sym_cfg.max_amount_monthly
                if sym_cfg.max_amount_monthly > 0
                else budget
            )
            while remainder >= px:
                _, px, shares, invested = rows[i]
                if invested + px > max_cap:
                    break
                shares += 1
                invested = round(shares * px, 2)
                remainder = round(remainder - px, 2)
                rows[i] = (sym_cfg, px, shares, invested)
            break

    lines: list[AllocationLine] = list(skip_lines)
    for index, (sym_cfg, px, shares, _) in enumerate(rows):
        rank = next(
            (i + 1 for i, s in enumerate(symbols) if s.symbol == sym_cfg.symbol),
            index + 1,
        )
        pending = evaluate_prescan_gate(sym_cfg.symbol, pmap.get(sym_cfg.symbol), gate)
        lines.append(
            _build_line(
                sym_cfg,
                shares,
                px,
                priority_rank=rank,
                note=pending.note if pending.note and not pending.blocked else None,
                avg_cost=costs.get(sym_cfg.symbol),
                high_3m=highs.get(sym_cfg.symbol),
                week52_high=w52.get(sym_cfg.symbol),
            )
        )
    order = {s.symbol: i for i, s in enumerate(symbols)}
    return tuple(sorted(lines, key=lambda row: order.get(row.symbol, 999)))


def equal_whole_share_lines(
    symbols: tuple[str, ...],
    budget: float,
    prices: dict[str, float | None],
    *,
    errors: dict[str, str] | None = None,
    highs_3m: dict[str, float | None] | None = None,
    monthly_per_symbol: dict[str, float] | None = None,
) -> tuple[AllocationLine, ...]:
    """Split *budget* across *symbols* with whole shares and roughly equal ₹."""
    err = errors or {}
    highs = highs_3m or {}
    priced: list[tuple[str, float]] = []
    lines: list[AllocationLine] = []
    for sym in symbols:
        if sym in err:
            lines.append(AllocationLine(sym, None, 0, 0.0, error=err[sym]))
            continue
        px = prices.get(sym)
        if px is None or px <= 0:
            lines.append(AllocationLine(sym, None, 0, 0.0, error="price unavailable"))
            continue
        priced.append((sym, px))

    if not priced:
        return tuple(lines)

    n = len(priced)
    target_each = budget / n
    alloc_rows: list[tuple[str, float, int, float]] = []
    for sym, px in priced:
        shares = max(1, int(target_each // px)) if px <= budget else 0
        alloc_rows.append((sym, px, shares, round(shares * px, 2)))

    total = sum(row[3] for row in alloc_rows)
    while total > budget:
        idx = max(range(len(alloc_rows)), key=lambda i: alloc_rows[i][3])
        sym, px, shares, invested = alloc_rows[idx]
        if shares <= 0:
            break
        shares -= 1
        alloc_rows[idx] = (sym, px, shares, round(shares * px, 2))
        total = sum(row[3] for row in alloc_rows)

    improved = True
    while improved:
        improved = False
        best: tuple[float, int] | None = None
        for i, (sym, px, shares, invested) in enumerate(alloc_rows):
            add = px
            if total + add <= budget:
                dev = abs((invested + add) - target_each)
                if best is None or dev < best[0]:
                    best = (dev, i)
        if best is None:
            break
        i = best[1]
        sym, px, shares, invested = alloc_rows[i]
        shares += 1
        invested = round(shares * px, 2)
        alloc_rows[i] = (sym, px, shares, invested)
        total += px
        improved = True

    monthly_lookup = monthly_per_symbol or {
        sym: (budget / n if n else 0.0) for sym, _ in priced
    }
    for sym, px, shares, invested in sorted(alloc_rows, key=lambda r: r[0]):
        dip_pct = dip_pct_from_high(px, highs.get(sym)) if sym in highs else None
        dip = classify_dip(px, highs.get(sym)) if sym in highs else None
        monthly_ref = monthly_lookup.get(sym, invested or target_each)
        topup = suggest_topup(dip, monthly_ref)
        lines.append(
            AllocationLine(
                symbol=sym,
                price=px,
                shares=shares,
                invested=invested,
                dip_pct=dip_pct,
                dip_label=dip,
                topup_range=topup,
            )
        )
    return tuple(sorted(lines, key=lambda row: row.symbol))


def priority_whole_share_lines(
    symbols: tuple[str, ...],
    budget: float,
    prices: dict[str, float | None],
    *,
    errors: dict[str, str] | None = None,
    highs_3m: dict[str, float | None] | None = None,
) -> tuple[AllocationLine, ...]:
    """Allocate by list order — first symbol is highest priority (P1)."""
    err = errors or {}
    highs = highs_3m or {}
    priority_ranks = {sym: index + 1 for index, sym in enumerate(symbols)}

    priced: list[tuple[str, float]] = []
    for sym in symbols:
        if sym in err:
            continue
        px = prices.get(sym)
        if px is None or px <= 0:
            continue
        priced.append((sym, px))

    share_map = dict.fromkeys(symbols, 0)
    spent = 0.0

    for sym, px in priced:
        if spent + px <= budget:
            share_map[sym] += 1
            spent += px

    improved = True
    while improved:
        improved = False
        for sym, px in priced:
            if spent + px <= budget:
                share_map[sym] += 1
                spent += px
                improved = True

    monthly_map = {
        sym: (share_map[sym] * px if share_map[sym] else px) for sym, px in priced
    }
    lines: list[AllocationLine] = []
    for index, sym in enumerate(symbols):
        rank = priority_ranks[sym]
        if sym in err:
            lines.append(
                AllocationLine(sym, None, 0, 0.0, priority_rank=rank, error=err[sym])
            )
            continue
        px = prices.get(sym)
        if px is None or px <= 0:
            lines.append(
                AllocationLine(
                    sym,
                    None,
                    0,
                    0.0,
                    priority_rank=rank,
                    error="price unavailable",
                )
            )
            continue
        shares = share_map.get(sym, 0)
        invested = round(shares * px, 2)
        dip_pct = dip_pct_from_high(px, highs.get(sym)) if sym in highs else None
        dip = classify_dip(px, highs.get(sym)) if sym in highs else None
        monthly_ref = monthly_map.get(sym, invested or px)
        topup = suggest_topup(dip, monthly_ref)
        lines.append(
            AllocationLine(
                symbol=sym,
                price=px,
                shares=shares,
                invested=invested,
                priority_rank=rank,
                dip_pct=dip_pct,
                dip_label=dip,
                topup_range=topup,
            )
        )
    return tuple(lines)


def allocate_portfolio(
    portfolio: PortfolioBucket,
    prices: dict[str, float | None],
    *,
    month: int,
    errors: dict[str, str] | None = None,
    highs_3m: dict[str, float | None] | None = None,
    week52_highs: dict[str, float | None] | None = None,
    avg_costs: dict[str, float | None] | None = None,
    carried_cash: float = 0.0,
    prescan_gate: PrescanGateConfig | None = None,
    prescan_map: dict[str, dict] | None = None,
) -> PortfolioAllocation:
    budget = round(portfolio.monthly_budget + max(carried_cash, 0.0), 2)
    mode = portfolio.allocation_mode
    gate = prescan_gate or PrescanGateConfig(enabled=False)
    pmap = prescan_map or {}

    symbols_for_alloc = portfolio.symbols
    if mode == "prescan_rank":
        symbols_for_alloc = rank_symbols_by_prescan(portfolio.symbols, pmap)
        mode = "equal_split"

    if mode == "priority":
        active_symbols: list[str] = []
        prescan_skip_lines: list[AllocationLine] = []
        for index, symbol in enumerate(portfolio.symbols):
            if not symbol.enabled:
                continue
            sym = symbol.symbol
            gate_result = evaluate_prescan_gate(sym, pmap.get(sym), gate)
            if gate_result.blocked:
                prescan_skip_lines.append(
                    _build_line(
                        symbol,
                        0,
                        prices.get(sym),
                        priority_rank=index + 1,
                        prescan_skip=True,
                        note=gate_result.note,
                        error=(errors or {}).get(sym),
                        avg_cost=(avg_costs or {}).get(sym),
                        high_3m=(highs_3m or {}).get(sym),
                        week52_high=(week52_highs or {}).get(sym),
                    )
                )
                continue
            active_symbols.append(sym)
        alloc_lines = priority_whole_share_lines(
            tuple(active_symbols),
            budget,
            prices,
            errors=errors,
            highs_3m=highs_3m,
        )
        lines = tuple(prescan_skip_lines) + alloc_lines
        order = {s.symbol: i for i, s in enumerate(portfolio.symbols)}
        lines = tuple(sorted(lines, key=lambda row: order.get(row.symbol, 999)))
    elif mode in {"equal_split", "equal"}:
        lines = target_split_whole_share_lines(
            symbols_for_alloc,
            budget,
            prices,
            month,
            errors=errors,
            highs_3m=highs_3m,
            week52_highs=week52_highs,
            avg_costs=avg_costs,
            overflow_symbol=portfolio.cash_policy.overflow_symbol,
            prescan_gate=gate,
            prescan_map=pmap,
        )
    else:
        lines = equal_whole_share_lines(
            symbol_names(portfolio),
            budget,
            prices,
            errors=errors,
            highs_3m=highs_3m,
        )

    invested = round(sum(line.invested for line in lines), 2)
    return PortfolioAllocation(
        portfolio=portfolio,
        lines=lines,
        invested=invested,
        cash_aside=round(budget - invested, 2),
    )


def _default_price_fetch(symbol: str) -> tuple[float | None, float | None, float | None]:
    try:
        data = fetch_price_data(symbol)
    except Exception as exc:  # noqa: BLE001 - resilience boundary per symbol fetch
        logger.warning("portfolio SIP price fetch failed for %s: %s", symbol, exc)
        return (None, None, None)
    high = three_month_high(data.ohlcv_adjusted)
    return (data.current_price_abs, high, data.week52_high_abs)


def fetch_prices_for_symbols(
    symbols: tuple[str, ...] | list[str],
    *,
    fetcher: PriceFetcher | None = None,
    max_workers: int = 6,
) -> tuple[
    dict[str, float | None],
    dict[str, float | None],
    dict[str, float | None],
    dict[str, str],
]:
    """Return (prices, highs_3m, week52_highs, errors) for each symbol."""
    fn = fetcher or _default_price_fetch
    unique = tuple(dict.fromkeys(s.upper() for s in symbols))
    prices: dict[str, float | None] = {}
    highs: dict[str, float | None] = {}
    week52: dict[str, float | None] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, sym): sym for sym in unique}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                price, high, w52 = future.result()
            except Exception as exc:  # noqa: BLE001 - resilience boundary per symbol fetch
                errors[sym] = str(exc)[:120]
                prices[sym] = None
                highs[sym] = None
                week52[sym] = None
                continue
            prices[sym] = price
            highs[sym] = high
            week52[sym] = w52
            if price is None:
                errors[sym] = "price unavailable"
    return prices, highs, week52, errors


def build_portfolio_sip_plan(
    config: PortfolioSipConfig | None = None,
    *,
    fetcher: PriceFetcher | None = None,
    month: int | None = None,
    avg_costs: dict[str, float | None] | None = None,
    path: Path | None = None,
) -> PortfolioSipPlan:
    cfg = config or load_portfolio_sip_config(path)
    if month is None:
        month = datetime.now(IST).month

    enabled_portfolios = tuple(p for p in cfg.portfolios if p.enabled)
    all_symbols = tuple(sym for p in enabled_portfolios for sym in symbol_names(p))
    prices, highs, week52, errors = fetch_prices_for_symbols(all_symbols, fetcher=fetcher)
    pmap = prescan_outcome_map()
    gate = cfg.prescan_gate
    allocations = tuple(
        allocate_portfolio(
            p,
            prices,
            month=month,
            errors=errors,
            highs_3m=highs,
            week52_highs=week52,
            avg_costs=avg_costs,
            prescan_gate=gate,
            prescan_map=pmap,
        )
        for p in enabled_portfolios
    )
    total_invested = round(sum(a.invested for a in allocations), 2)
    total_cash = round(sum(a.cash_aside for a in allocations), 2)
    return PortfolioSipPlan(
        config=cfg,
        allocations=allocations,
        total_invested=total_invested,
        total_cash_aside=total_cash,
    )


__all__ = [
    "AllocationLine",
    "PortfolioAllocation",
    "PortfolioBucket",
    "PortfolioConfig",
    "PortfolioSipConfig",
    "PortfolioSipPlan",
    "allocate_portfolio",
    "build_portfolio_sip_plan",
    "load_portfolio_sip_config",
]
