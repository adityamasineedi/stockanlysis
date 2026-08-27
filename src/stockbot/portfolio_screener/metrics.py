"""Extract flat StockMetrics from existing fetch-layer objects.

Never invents values. Missing → None + missing reason."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from stockbot.models import Financials, PriceData, Shareholding, TickerInfo
from stockbot.portfolio_screener.models import StockMetrics
from stockbot.portfolio_screener.score_utils import (
    cagr,
    latest,
    safe_div,
    series_present,
)

logger = logging.getLogger(__name__)

# Screener row aliases — first match wins.
_REVENUE_ALIASES = ("Sales", "Revenue", "Total Income")
_OP_PROFIT_ALIASES = ("Operating Profit", "OPM")
_EBITDA_ALIASES = ("OPM %",)  # margin only — EBITDA level often = Operating Profit on Screener
_INTEREST_ALIASES = ("Interest", "Finance Cost", "Finance Costs")
_DEPRECIATION_ALIASES = ("Depreciation",)
_PAT_ALIASES = ("Net Profit", "Profit after Tax", "PAT")
_EPS_ALIASES = ("EPS in Rs", "EPS", "Earning Per Share")
_OCF_ALIASES = ("Cash from Operating Activity", "Cash from Operating Activities")
_CAPEX_ALIASES = ("Cash from Investing Activity", "Cash from Investing Activities")
_FCF_ALIASES = ("Free Cash Flow",)  # rarely present as its own row
_BORROWINGS_ALIASES = ("Borrowings", "Total Borrowings", "Debt")
_EQUITY_ALIASES = ("Equity Capital", "Share Capital")
_RESERVES_ALIASES = ("Reserves",)
_CASH_ALIASES = ("Cash Equivalents", "Cash", "Cash and Bank")
_TOTAL_ASSETS_ALIASES = ("Total Assets",)
_ROE_ALIASES = ("ROE %", "ROE")
_ROCE_ALIASES = ("ROCE %", "ROCE")
_CURRENT_RATIO_ALIASES = ("Current Ratio",)
_DEBTOR_DAYS_ALIASES = ("Debtor Days", "Debtor days")


def _row(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series | None:
    for name in aliases:
        if name in df.index:
            return df.loc[name]
    return None


def _series_values(row: pd.Series | None) -> list[float | None]:
    if row is None:
        return []
    out: list[float | None] = []
    for v in row.tolist():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(None)
    return out


def _mark_missing(metrics: StockMetrics, field: str, reason: str) -> None:
    metrics.missing[field] = reason


def _cagr_from_series(values: list[float | None], years: int) -> float | None:
    present = series_present(values)
    if len(present) < years + 1:
        # try with whatever length we have if at least 2 points spanning ~years
        if len(present) < 2:
            return None
        span = len(present) - 1
        if span < max(2, years - 1):
            return None
        return cagr(present[0], present[-1], float(span))
    window = present[-(years + 1) :]
    return cagr(window[0], window[-1], float(years))


def _compute_interest_coverage(
    operating_profit: float | None,
    interest: float | None,
) -> float | None:
    if operating_profit is None or interest is None:
        return None
    if interest <= 0:
        return 100.0 if operating_profit > 0 else None
    return operating_profit / interest


def _compute_margins(
    revenue: float | None,
    operating_profit: float | None,
) -> float | None:
    return safe_div(operating_profit, revenue)


def fetch_market_metadata(symbol: str) -> dict[str, float | str | None]:
    """Cheap yfinance info pull for sector / market cap / forward PE.

    Failures are non-fatal — returns Nones rather than raising.
    """
    meta: dict[str, float | str | None] = {
        "sector": None,
        "industry": None,
        "market_cap_cr": None,
        "forward_pe": None,
        "trailing_pe": None,
        "pb": None,
        "dividend_yield_pct": None,
        "shares_outstanding": None,
    }
    for suffix in (".NS", ".BO"):
        try:
            info = yf.Ticker(f"{symbol}{suffix}").info or {}
        except Exception as exc:  # noqa: BLE001 — metadata is optional
            logger.info("yfinance info failed for %s%s: %s", symbol, suffix, exc)
            continue
        if not info:
            continue
        market_cap = info.get("marketCap")
        if market_cap:
            meta["market_cap_cr"] = float(market_cap) / 1e7  # ₹ → crore
        meta["sector"] = info.get("sector") or meta["sector"]
        meta["industry"] = info.get("industry") or meta["industry"]
        for src, dest in (
            ("forwardPE", "forward_pe"),
            ("trailingPE", "trailing_pe"),
            ("priceToBook", "pb"),
        ):
            raw = info.get(src)
            if raw is not None:
                try:
                    meta[dest] = float(raw)
                except (TypeError, ValueError):
                    pass
        dy = info.get("dividendYield")
        if dy is not None:
            try:
                # yfinance may return fraction or percent depending on version
                dy_f = float(dy)
                meta["dividend_yield_pct"] = dy_f * 100.0 if dy_f < 1.0 else dy_f
            except (TypeError, ValueError):
                pass
        shares = info.get("sharesOutstanding")
        if shares:
            try:
                meta["shares_outstanding"] = float(shares)
            except (TypeError, ValueError):
                pass
        if meta["sector"] or meta["market_cap_cr"]:
            break
    return meta


def extract_price_returns(price: PriceData, lookback_days: int = 252) -> list[float]:
    closes = price.ohlcv_adjusted["Close"].dropna()
    if len(closes) < 30:
        return []
    window = closes.iloc[-lookback_days:] if len(closes) > lookback_days else closes
    rets = window.pct_change().dropna()
    return [float(x) for x in rets.tolist()]


def extract_metrics(
    ticker: TickerInfo,
    *,
    financials: Financials | None,
    price: PriceData | None,
    shareholding: Shareholding | None,
    market_meta: dict[str, float | str | None] | None = None,
) -> StockMetrics:
    m = StockMetrics(
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        data_timestamp=datetime.now(UTC),
    )

    if price is None:
        _mark_missing(m, "current_price_abs", "price fetch failed")
    else:
        m.current_price_abs = price.current_price_abs
        m.price_returns = extract_price_returns(price)

    meta = market_meta if market_meta is not None else {}
    sector = meta.get("sector")
    industry = meta.get("industry")
    m.sector = str(sector) if sector else "Unknown"
    m.industry = str(industry) if industry else "Unknown"
    if sector is None:
        _mark_missing(m, "sector", "sector unavailable — defaulted to Unknown")
    if industry is None:
        _mark_missing(m, "industry", "industry unavailable — defaulted to Unknown")

    mcap = meta.get("market_cap_cr")
    m.market_cap_cr = float(mcap) if isinstance(mcap, (int, float)) else None
    if m.market_cap_cr is None:
        _mark_missing(m, "market_cap_cr", "market cap unavailable")

    for key, attr in (
        ("forward_pe", "forward_pe"),
        ("pb", "pb"),
        ("dividend_yield_pct", "dividend_yield_pct"),
    ):
        raw = meta.get(key)
        if isinstance(raw, (int, float)):
            setattr(m, attr, float(raw))
        else:
            _mark_missing(m, attr, f"{attr} unavailable from market metadata")

    trailing_pe = meta.get("trailing_pe")
    if isinstance(trailing_pe, (int, float)):
        m.pe = float(trailing_pe)

    if shareholding is None:
        _mark_missing(m, "promoter_pct", "shareholding fetch failed")
        _mark_missing(m, "promoter_pledge_pct", "shareholding fetch failed")
    else:
        m.promoter_pct = shareholding.promoter_pct
        m.promoter_pledge_pct = shareholding.pledge_pct_of_promoter_holding
        if m.promoter_pct is None:
            _mark_missing(m, "promoter_pct", "promoter holding not reported")
        if m.promoter_pledge_pct is None:
            _mark_missing(m, "promoter_pledge_pct", "pledge status unconfirmed")

    if financials is None:
        for field in (
            "revenue",
            "net_income",
            "eps",
            "operating_cash_flow",
            "free_cash_flow",
            "roe",
            "roce",
            "debt",
            "ebit",
            "ebitda",
        ):
            _mark_missing(m, field, "fundamentals fetch failed")
        return m

    m.years_available = financials.years_available
    pnl = financials.pnl
    bs = financials.balance_sheet
    cf = financials.cash_flow
    ratios = financials.ratios

    rev_row = _row(pnl, _REVENUE_ALIASES)
    m.revenue_series = _series_values(rev_row)
    m.revenue = latest(m.revenue_series)
    if m.revenue is None:
        _mark_missing(m, "revenue", "Sales/Revenue row missing")

    op_row = _row(pnl, _OP_PROFIT_ALIASES)
    op_series = _series_values(op_row)
    m.operating_profit = latest(op_series)
    m.ebit = m.operating_profit  # Screener Operating Profit ≈ EBIT for non-banks
    if m.operating_profit is None:
        _mark_missing(m, "ebit", "Operating Profit row missing")
        _mark_missing(m, "operating_profit", "Operating Profit row missing")

    # Screener rarely has a clean EBITDA line; approximate as OP + Depreciation.
    dep_series = _series_values(_row(pnl, _DEPRECIATION_ALIASES))
    if m.operating_profit is not None and latest(dep_series) is not None:
        m.ebitda = m.operating_profit + float(latest(dep_series))
        m.ebitda_series = [
            (o + d) if o is not None and d is not None else None
            for o, d in zip(op_series, dep_series, strict=False)
        ]
    elif m.operating_profit is not None:
        m.ebitda = m.operating_profit
        m.ebitda_series = list(op_series)
        m.raw_notes.append("EBITDA approximated as Operating Profit (depreciation missing)")
    else:
        _mark_missing(m, "ebitda", "cannot compute EBITDA without operating profit")

    pat_row = _row(pnl, _PAT_ALIASES)
    m.net_income_series = _series_values(pat_row)
    m.net_income = latest(m.net_income_series)
    if m.net_income is None:
        _mark_missing(m, "net_income", "Net Profit row missing")

    eps_row = _row(pnl, _EPS_ALIASES)
    m.eps_series = _series_values(eps_row)
    m.eps = latest(m.eps_series)
    if m.eps is None:
        _mark_missing(m, "eps", "EPS row missing")

    interest = latest(_series_values(_row(pnl, _INTEREST_ALIASES)))
    m.interest_coverage = _compute_interest_coverage(m.operating_profit, interest)
    if m.interest_coverage is None:
        _mark_missing(m, "interest_coverage", "interest and/or operating profit missing")

    ocf_row = _row(cf, _OCF_ALIASES)
    m.ocf_series = _series_values(ocf_row)
    m.operating_cash_flow = latest(m.ocf_series)
    if m.operating_cash_flow is None:
        _mark_missing(m, "operating_cash_flow", "Cash from Operating Activity missing")

    # FCF ≈ OCF + Cash from Investing (investing is typically negative = capex outflow)
    invest_series = _series_values(_row(cf, _CAPEX_ALIASES))
    fcf_direct = _series_values(_row(cf, _FCF_ALIASES))
    if series_present(fcf_direct):
        m.fcf_series = fcf_direct
        m.free_cash_flow = latest(fcf_direct)
    elif m.ocf_series and invest_series:
        m.fcf_series = [
            (o + i) if o is not None and i is not None else None
            for o, i in zip(m.ocf_series, invest_series, strict=False)
        ]
        m.free_cash_flow = latest(m.fcf_series)
        m.raw_notes.append("FCF approximated as OCF + Cash from Investing")
    else:
        _mark_missing(m, "free_cash_flow", "cannot compute FCF")

    debt_row = _row(bs, _BORROWINGS_ALIASES)
    m.debt_series = _series_values(debt_row)
    m.debt = latest(m.debt_series)
    if m.debt is None:
        # Zero borrowings sometimes means the row is absent for debt-free cos.
        # Treat missing as unavailable, not zero.
        _mark_missing(m, "debt", "Borrowings row missing")

    cash_row = _row(bs, _CASH_ALIASES)
    m.cash = latest(_series_values(cash_row))
    if m.cash is None:
        _mark_missing(m, "cash", "Cash Equivalents row missing (often nested under Other Assets)")

    if m.debt is not None and m.cash is not None:
        m.net_debt = m.debt - m.cash
    elif m.debt is not None:
        m.net_debt = m.debt
        m.raw_notes.append("net_debt uses gross debt (cash unavailable)")
    else:
        _mark_missing(m, "net_debt", "debt and/or cash unavailable")

    equity_cap = latest(_series_values(_row(bs, _EQUITY_ALIASES)))
    reserves = latest(_series_values(_row(bs, _RESERVES_ALIASES)))
    if equity_cap is not None and reserves is not None:
        m.equity = equity_cap + reserves
    elif reserves is not None:
        m.equity = reserves
    else:
        _mark_missing(m, "equity", "Equity Capital / Reserves missing")

    m.debt_equity = safe_div(m.debt, m.equity) if m.debt is not None and m.equity else None
    if m.debt is not None and m.equity is not None and m.equity <= 0:
        m.debt_equity = None
        m.raw_notes.append("negative/zero equity — debt_equity undefined")
        _mark_missing(m, "debt_equity", "non-positive equity")

    m.net_debt_ebitda = safe_div(m.net_debt, m.ebitda) if m.ebitda and m.ebitda > 0 else None

    m.roe_series = _series_values(_row(ratios, _ROE_ALIASES))
    m.roe = latest(m.roe_series)
    if m.roe is None:
        _mark_missing(m, "roe", "ROE % missing from ratios")

    m.roce_series = _series_values(_row(ratios, _ROCE_ALIASES))
    m.roce = latest(m.roce_series)
    if m.roce is None:
        _mark_missing(m, "roce", "ROCE % missing from ratios")

    m.current_ratio = latest(_series_values(_row(ratios, _CURRENT_RATIO_ALIASES)))
    if m.current_ratio is None:
        _mark_missing(m, "current_ratio", "Current Ratio missing")

    m.operating_margin = _compute_margins(m.revenue, m.operating_profit)
    m.ebitda_margin = _compute_margins(m.revenue, m.ebitda)
    if rev_row is not None and op_row is not None:
        m.operating_margin_series = [
            safe_div(o, r) for o, r in zip(op_series, m.revenue_series, strict=False)
        ]

    total_assets = latest(_series_values(_row(bs, _TOTAL_ASSETS_ALIASES)))
    m.asset_turnover = safe_div(m.revenue, total_assets)

    m.ocf_to_pat = safe_div(m.operating_cash_flow, m.net_income) if m.net_income else None
    m.fcf_to_pat = safe_div(m.free_cash_flow, m.net_income) if m.net_income else None
    m.fcf_margin = safe_div(m.free_cash_flow, m.revenue)

    # Dilution proxy: equity capital trend
    eq_series = _series_values(_row(bs, _EQUITY_ALIASES))
    present_eq = series_present(eq_series)
    if len(present_eq) >= 2 and present_eq[0] > 0:
        m.share_dilution_pct = ((present_eq[-1] / present_eq[0]) - 1.0) * 100.0
    else:
        _mark_missing(m, "share_dilution_pct", "insufficient equity capital history")

    m.revenue_cagr_3y = _cagr_from_series(m.revenue_series, 3)
    m.revenue_cagr_5y = _cagr_from_series(m.revenue_series, 5)
    m.eps_cagr_3y = _cagr_from_series(m.eps_series, 3)
    m.eps_cagr_5y = _cagr_from_series(m.eps_series, 5)
    m.ebitda_cagr_3y = _cagr_from_series(m.ebitda_series, 3)

    if m.pe is None and m.current_price_abs and m.eps and m.eps > 0:
        m.pe = m.current_price_abs / m.eps
    elif m.pe is None:
        _mark_missing(m, "pe", "cannot compute P/E (price or positive EPS missing)")

    if m.peg is None and m.pe is not None and m.eps_cagr_3y is not None and m.eps_cagr_3y > 0:
        m.peg = m.pe / (m.eps_cagr_3y * 100.0)
    else:
        _mark_missing(m, "peg", "PEG requires P/E and positive EPS CAGR")

    if m.market_cap_cr is not None and m.free_cash_flow and m.free_cash_flow > 0:
        m.price_fcf = m.market_cap_cr / m.free_cash_flow
    else:
        _mark_missing(m, "price_fcf", "Price/FCF requires market cap and positive FCF")

    if m.market_cap_cr is not None and m.net_debt is not None and m.ebitda and m.ebitda > 0:
        enterprise_value = m.market_cap_cr + m.net_debt
        m.ev_ebitda = enterprise_value / m.ebitda
    else:
        _mark_missing(m, "ev_ebitda", "EV/EBITDA requires market cap, net debt, positive EBITDA")

    # Historical valuation percentile intentionally left unavailable —
    # fabricating it is forbidden.
    _mark_missing(
        m,
        "valuation_percentile_historical",
        "historical valuation series not available in current data sources",
    )

    return m
