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
_EQUITY_ALIASES = (
    "Equity Capital",
    "Share Capital",
    "Equity Share Capital",
)
_RESERVES_ALIASES = (
    "Reserves",
    "Reserves and Surplus",
    "Other Equity",
)
_TOTAL_EQUITY_ALIASES = (
    "Total Equity",
    "Shareholders Funds",
    "Shareholder's Funds",
    "Shareholders' Funds",
    "Net Worth",
    "Total Shareholders Funds",
)
_CASH_ALIASES = ("Cash Equivalents", "Cash", "Cash and Bank")
_TOTAL_ASSETS_ALIASES = ("Total Assets",)
_ROE_ALIASES = ("ROE %", "ROE")
_ROCE_ALIASES = ("ROCE %", "ROCE")
_CURRENT_RATIO_ALIASES = ("Current Ratio",)
_CURRENT_ASSETS_ALIASES = ("Current Assets", "Total Current Assets")
_CURRENT_LIAB_ALIASES = ("Current Liabilities", "Total Current Liabilities")
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


def _series_values_excluding_ttm(row: pd.Series | None) -> list[float | None]:
    """Like _series_values but drops a trailing "TTM" column by label.

    Screener's P&L table often carries a TTM column that its cash-flow table
    does not. Positionally pairing "last N" tails of two independently-parsed
    series can then silently sum different fiscal periods. Use this only where
    a same-fiscal-year alignment across statements is required (e.g. cumulative
    multi-year OCF/PAT ratios) — not for "latest value" fields, where TTM is the
    most current and desired figure.
    """
    if row is None:
        return []
    out: list[float | None] = []
    for col, v in row.items():
        if str(col).strip().upper() == "TTM":
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(None)
    return out


# yfinance sector/industry mislabels that break issuer routing (e.g. V-Guard → Utilities).
_TICKER_SECTOR_OVERRIDES: dict[str, tuple[str, str]] = {
    "VGUARD": ("Consumer Cyclical", "Consumer Electronics"),
    "SERVOTECH": ("Industrials", "Electrical Equipment & Instruments"),
}

_KEY_RATIO_FIELDS = frozenset(
    {"roe", "roce", "debt_equity", "ocf_to_pat", "interest_coverage", "net_debt_ebitda"}
)


def count_derived_key_ratios(metrics: StockMetrics) -> int:
    """How many gatekeeper ratios were computed/yfinance rather than Screener-fetched."""
    return sum(
        1
        for field in _KEY_RATIO_FIELDS
        if metrics.metric_sources.get(field) in ("computed", "yfinance")
    )


def _mark_missing(metrics: StockMetrics, field: str, reason: str) -> None:
    metrics.missing[field] = reason


def _clamp_pct(value: float | None) -> float | None:
    """Accept only 0–100 percentage points; reject garbage before scoring."""
    if value is None:
        return None
    if value < 0.0 or value > 100.0:
        logger.warning("rejecting out-of-range percentage value: %s", value)
        return None
    return float(value)


def _set_metric(
    metrics: StockMetrics,
    field: str,
    value: float | None,
    *,
    source: str,
    missing_reason: str,
) -> None:
    """Set a metric value and record its provenance; clear missing if filled."""
    if value is None:
        _mark_missing(metrics, field, missing_reason)
        return
    setattr(metrics, field, value)
    metrics.metric_sources[field] = source
    metrics.missing.pop(field, None)


def _equity_book_series(bs: pd.DataFrame) -> list[float | None]:
    """Book equity series: Total Equity row, else Equity Capital + Reserves."""
    total = _series_values(_row(bs, _TOTAL_EQUITY_ALIASES))
    if any(v is not None for v in total):
        return total

    eq = _series_values(_row(bs, _EQUITY_ALIASES))
    res = _series_values(_row(bs, _RESERVES_ALIASES))
    n = max(len(eq), len(res))
    out: list[float | None] = []
    for i in range(n):
        e = eq[i] if i < len(eq) else None
        r = res[i] if i < len(res) else None
        if e is not None and r is not None:
            out.append(e + r)
        elif r is not None:
            out.append(r)
        elif e is not None:
            out.append(e)
        else:
            out.append(None)
    return out


def _roe_series_from_pat_equity(
    pat: list[float | None], equity: list[float | None]
) -> list[float | None]:
    n = min(len(pat), len(equity))
    out: list[float | None] = []
    for i in range(n):
        p, e = pat[i], equity[i]
        if p is None or e is None or e == 0:
            out.append(None)
        else:
            out.append((p / e) * 100.0)
    return out


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
        "roe_pct": None,
        "roce_pct": None,
        "analyst_count": None,
        "recommendation_key": None,
        "target_mean_price": None,
        "target_low_price": None,
        "target_high_price": None,
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
        for yf_key, meta_key in (
            ("returnOnEquity", "roe_pct"),
            ("returnOnCapital", "roce_pct"),
        ):
            if meta[meta_key] is not None:
                continue
            raw = info.get(yf_key)
            if raw is None:
                continue
            try:
                val = float(raw)
                meta[meta_key] = val * 100.0 if abs(val) <= 2.0 else val
            except (TypeError, ValueError):
                pass
        for yf_key, meta_key in (
            ("numberOfAnalystOpinions", "analyst_count"),
            ("recommendationKey", "recommendation_key"),
            ("targetMeanPrice", "target_mean_price"),
            ("targetLowPrice", "target_low_price"),
            ("targetHighPrice", "target_high_price"),
        ):
            raw = info.get(yf_key)
            if raw is None:
                continue
            if meta_key == "analyst_count":
                try:
                    meta[meta_key] = int(raw)
                except (TypeError, ValueError):
                    pass
            elif meta_key == "recommendation_key":
                meta[meta_key] = str(raw)
            else:
                try:
                    meta[meta_key] = float(raw)
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
    sector_override = _TICKER_SECTOR_OVERRIDES.get(ticker.symbol.upper())
    if sector_override is not None:
        m.sector, m.industry = sector_override
        m.sector_source = "override"
        m.raw_notes.append(
            f"Sector/industry override for {ticker.symbol}: {m.sector} / {m.industry}"
        )
    else:
        m.sector = str(sector) if sector else "Unknown"
        m.industry = str(industry) if industry else "Unknown"
        m.sector_source = "yfinance" if sector else "unknown"
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
        _mark_missing(m, "promoter_holding_pct", "shareholding fetch failed")
        _mark_missing(m, "pledged_promoter_holding_pct", "shareholding fetch failed")
    else:
        # Map fetch-layer Shareholding fields → explicit screener names so
        # holding % is never confused with pledge % of promoter holding.
        holding = _clamp_pct(shareholding.promoter_pct)
        pledge = _clamp_pct(shareholding.pledge_pct_of_promoter_holding)
        m.promoter_holding_pct = holding
        m.pledged_promoter_holding_pct = pledge
        if holding is None:
            _mark_missing(m, "promoter_holding_pct", "promoter holding not reported")
        else:
            m.metric_sources["promoter_holding_pct"] = "fetched"
        if pledge is None:
            _mark_missing(
                m,
                "pledged_promoter_holding_pct",
                "pledge status unconfirmed (not the same as promoter holding %)",
            )
        else:
            m.metric_sources["pledged_promoter_holding_pct"] = "fetched"

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
    m.financials_basis = financials.basis
    m.financials_source = financials.source
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
    m.net_income_series_fy_only = _series_values_excluding_ttm(pat_row)
    m.net_income = latest(m.net_income_series)
    if m.net_income is None:
        _mark_missing(m, "net_income", "Net Profit row missing")

    eps_row = _row(pnl, _EPS_ALIASES)
    m.eps_series = _series_values(eps_row)
    m.eps = latest(m.eps_series)
    if m.eps is None:
        _mark_missing(m, "eps", "EPS row missing")

    interest = latest(_series_values(_row(pnl, _INTEREST_ALIASES)))
    cov = _compute_interest_coverage(m.operating_profit, interest)
    if cov is not None:
        _set_metric(
            m,
            "interest_coverage",
            cov,
            source="computed",
            missing_reason="interest coverage unavailable",
        )
        m.raw_notes.append(
            f"Interest coverage computed as Operating Profit / Interest = {cov:.2f}"
        )
    else:
        _mark_missing(m, "interest_coverage", "interest and/or operating profit missing")

    ocf_row = _row(cf, _OCF_ALIASES)
    m.ocf_series = _series_values(ocf_row)
    m.ocf_series_fy_only = _series_values_excluding_ttm(ocf_row)
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

    equity_series = _equity_book_series(bs)
    equity_cap = latest(_series_values(_row(bs, _EQUITY_ALIASES)))
    reserves = latest(_series_values(_row(bs, _RESERVES_ALIASES)))
    m.equity = latest(equity_series)
    if m.equity is None:
        if equity_cap is not None and reserves is not None:
            m.equity = equity_cap + reserves
        elif reserves is not None:
            m.equity = reserves
        else:
            _mark_missing(m, "equity", "Equity Capital / Reserves missing")

    if m.debt is not None and m.equity is not None and m.equity > 0:
        _set_metric(
            m,
            "debt_equity",
            safe_div(m.debt, m.equity),
            source="computed",
            missing_reason="cannot compute debt_equity",
        )
    elif m.debt is not None and m.equity is not None and m.equity <= 0:
        m.debt_equity = None
        m.raw_notes.append("negative/zero equity — debt_equity undefined")
        _mark_missing(m, "debt_equity", "non-positive equity")
    else:
        _mark_missing(m, "debt_equity", "debt and/or equity unavailable")

    if m.net_debt is not None and m.ebitda is not None and m.ebitda > 0:
        _set_metric(
            m,
            "net_debt_ebitda",
            safe_div(m.net_debt, m.ebitda),
            source="computed",
            missing_reason="net_debt_ebitda unavailable",
        )
    else:
        m.net_debt_ebitda = None
        if m.net_debt is None or m.ebitda is None or m.ebitda <= 0:
            _mark_missing(m, "net_debt_ebitda", "net debt and/or positive EBITDA missing")

    # ROE: Screener ratios → compute from PAT/book equity → point fallback → yfinance
    m.roe_series = _series_values(_row(ratios, _ROE_ALIASES))
    m.roe = latest(m.roe_series)
    if m.roe is not None:
        m.metric_sources["roe"] = "fetched"
        m.missing.pop("roe", None)
    else:
        computed_series = _roe_series_from_pat_equity(m.net_income_series, equity_series)
        computed_roe = latest(computed_series)
        if computed_roe is None and m.net_income is not None and m.equity not in (None, 0):
            computed_roe = (m.net_income / m.equity) * 100.0
            computed_series = [*computed_series, computed_roe]
        if computed_roe is not None:
            m.roe_series = computed_series
            _set_metric(
                m,
                "roe",
                computed_roe,
                source="computed",
                missing_reason="ROE unavailable",
            )
            m.raw_notes.append(
                f"ROE % computed from Net Profit / book equity = {computed_roe:.2f}"
            )
        else:
            yf_roe = meta.get("roe_pct") if meta else None
            if isinstance(yf_roe, (int, float)):
                _set_metric(
                    m,
                    "roe",
                    float(yf_roe),
                    source="yfinance",
                    missing_reason="ROE unavailable",
                )
                m.raw_notes.append(f"ROE % from yfinance returnOnEquity = {float(yf_roe):.2f}")
            else:
                _mark_missing(m, "roe", "ROE % missing from ratios; cannot compute from P&L+BS")

    # ROCE: Screener ratios → EBIT / (Equity + Debt) → yfinance
    m.roce_series = _series_values(_row(ratios, _ROCE_ALIASES))
    m.roce = latest(m.roce_series)
    if m.roce is not None:
        m.metric_sources["roce"] = "fetched"
        m.missing.pop("roce", None)
    else:
        capital_employed = None
        if m.equity is not None:
            capital_employed = m.equity + (m.debt or 0.0)
        computed_roce = None
        if (
            m.operating_profit is not None
            and capital_employed is not None
            and capital_employed > 0
        ):
            computed_roce = (m.operating_profit / capital_employed) * 100.0
        if computed_roce is not None:
            _set_metric(
                m,
                "roce",
                computed_roce,
                source="computed",
                missing_reason="ROCE unavailable",
            )
            m.raw_notes.append(
                "ROCE % computed from Operating Profit / "
                f"(Equity Capital + Reserves + Borrowings) = {computed_roce:.2f}"
            )
        else:
            yf_roce = meta.get("roce_pct") if meta else None
            if isinstance(yf_roce, (int, float)):
                _set_metric(
                    m,
                    "roce",
                    float(yf_roce),
                    source="yfinance",
                    missing_reason="ROCE unavailable",
                )
                m.raw_notes.append(f"ROCE % from yfinance = {float(yf_roce):.2f}")
            else:
                _mark_missing(
                    m, "roce", "ROCE % missing from ratios; cannot compute from P&L+BS"
                )

    m.current_ratio = latest(_series_values(_row(ratios, _CURRENT_RATIO_ALIASES)))
    if m.current_ratio is not None:
        m.metric_sources["current_ratio"] = "fetched"
        m.missing.pop("current_ratio", None)
    else:
        ca = latest(_series_values(_row(bs, _CURRENT_ASSETS_ALIASES)))
        cl = latest(_series_values(_row(bs, _CURRENT_LIAB_ALIASES)))
        computed_cr = safe_div(ca, cl) if ca is not None and cl is not None else None
        if computed_cr is not None:
            _set_metric(
                m,
                "current_ratio",
                computed_cr,
                source="computed",
                missing_reason="Current Ratio unavailable",
            )
            m.raw_notes.append(
                f"Current Ratio computed from Current Assets / Current Liabilities = {computed_cr:.2f}"
            )
        else:
            _mark_missing(
                m,
                "current_ratio",
                "Current Ratio missing; Current Assets/Liabilities not on condensed BS",
            )

    m.operating_margin = _compute_margins(m.revenue, m.operating_profit)
    if m.operating_margin is not None:
        m.metric_sources["operating_margin"] = "computed"
    m.ebitda_margin = _compute_margins(m.revenue, m.ebitda)
    if m.ebitda_margin is not None:
        m.metric_sources["ebitda_margin"] = "computed"
    if rev_row is not None and op_row is not None:
        m.operating_margin_series = [
            safe_div(o, r) for o, r in zip(op_series, m.revenue_series, strict=False)
        ]

    total_assets = latest(_series_values(_row(bs, _TOTAL_ASSETS_ALIASES)))
    m.asset_turnover = safe_div(m.revenue, total_assets)
    if m.asset_turnover is not None:
        m.metric_sources["asset_turnover"] = "computed"

    m.ocf_to_pat = safe_div(m.operating_cash_flow, m.net_income) if m.net_income else None
    if m.ocf_to_pat is not None:
        m.metric_sources["ocf_to_pat"] = "computed"
        m.missing.pop("ocf_to_pat", None)
    else:
        _mark_missing(m, "ocf_to_pat", "OCF and/or Net Profit unavailable")
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
