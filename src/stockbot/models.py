"""Data contracts for stockbot. See PROJECT.md — this module implements
exactly what's documented there; the two must never drift apart."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class TickerInfo:
    symbol: str
    exchange: Literal["NSE", "BSE"]
    company_name: str
    isin: str | None


@dataclass(frozen=True)
class AmbiguousMatch:
    candidates: list[TickerInfo]
    scores: list[float]


@dataclass(frozen=True, eq=False)
class PriceData:
    current_price_abs: float
    price_date: date
    ohlcv_adjusted: pd.DataFrame
    ohlcv_unadjusted: pd.DataFrame
    week52_high_abs: float
    week52_low_abs: float
    source: str
    fetched_at: datetime


@dataclass(frozen=True)
class Technicals:
    sma50: float | None
    sma200: float | None
    rsi14: float | None
    support_abs: list[float]
    resistance_abs: list[float]
    as_of_date: date
    source: str
    fetched_at: datetime
    bollinger_mid: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    bollinger_bandwidth_pct: float | None = None
    price_vs_bollinger: str | None = None
    trend_label: str | None = None


@dataclass(frozen=True, eq=False)
class Financials:
    pnl: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    ratios: pd.DataFrame
    quarterly: pd.DataFrame
    basis: Literal["consolidated", "standalone"]
    years_available: int
    source: str
    fetched_at: datetime
    # Not a financial figure — lives here because it's scraped from the
    # same Screener page fetch (its "About" block) rather than costing a
    # separate request. None if Screener has no About block for this
    # company; a genuine MISSING, not an empty-string default.
    business_description: str | None = None


@dataclass(frozen=True)
class Shareholding:
    promoter_pct: float | None
    pledge_pct_of_promoter_holding: float | None
    fii_pct: float | None
    dii_pct: float | None
    quarter: str | None
    source: Literal["NSE", "BSE", "Screener"]
    fetched_at: datetime


@dataclass(frozen=True)
class RedFlag:
    headline: str
    url: str
    published_date: date
    found_by_query: str


@dataclass(frozen=True)
class NewsItems:
    general: list[RedFlag]
    red_flags: list[RedFlag]
    queries_run: list[str]
    queries_empty: list[str]
    source: str
    fetched_at: datetime


@dataclass(frozen=True)
class ArBusinessSummary:
    """Rule-parsed highlights from MD&A / order-book / segment AR sections."""

    segments: tuple[str, ...] = ()
    order_book_cr: float | None = None
    order_book_horizon_years: float | None = None
    key_risks: tuple[str, ...] = ()
    strategy: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportText:
    sections: dict[str, str]
    report_year: int | None
    source_url: str | None
    truncated: bool
    dropped_sections: list[str]
    source: str
    fetched_at: datetime
    business_summary: ArBusinessSummary | None = None


@dataclass(frozen=True)
class BriefMetadata:
    ticker: str
    company_name: str
    sector: str | None
    industry: str | None
    market_cap_cr: float | None
    ttm_pe: float | None  # Yahoo's own trailingPE snapshot — secondary reference only
    ttm_pb: float | None
    price: float
    price_date: str
    range_52w_low: float
    range_52w_high: float
    rsi_14: float | None
    ttm_eps: float | None = None  # TTM EPS read from the same FINANCIALS table as the report
    pe_price_eps: float | None = None  # price / ttm_eps — the multiple the report must quote
    adv_inr_cr: float | None = None  # average daily turnover ₹ crore
    quarterly_pe: float | None = None  # price / (latest quarter EPS × 4)
    sales_ttm_cr: float | None = None
    pat_ttm_cr: float | None = None
    opm_pct: float | None = None
    order_book_cr: float | None = None


@dataclass(frozen=True)
class StreetConsensus:
    """yfinance analyst aggregate — tension diagnostic only, not thesis input."""

    source: str
    analyst_count: int | None
    recommendation_key: str | None
    target_mean_price: float | None
    target_low_price: float | None
    target_high_price: float | None
    price_vs_target_pct: float | None
    tension: str
    note: str


@dataclass(frozen=True)
class PeerRow:
    symbol: str
    pe: float | None
    roe_pct: float | None
    market_cap_cr: float | None


@dataclass(frozen=True)
class PeerSnapshot:
    target_symbol: str
    sector: str | None
    target_pe: float | None
    target_roe_pct: float | None
    peer_median_pe: float | None
    peer_count: int
    pe_percentile: float | None
    sector_benchmark_pe_fair: float | None
    peers: tuple[PeerRow, ...]
    note: str | None = None


@dataclass(frozen=True)
class SectorScorecardContext:
    issuer_class: str | None
    scorecard_lens: str
    supplied_metrics: tuple[tuple[str, str], ...]
    ar_snippets: tuple[str, ...]
    generic_quant_note: str | None = None


@dataclass(frozen=True)
class PortfolioExecutionContext:
    in_sip_portfolio: bool
    sip_bucket: str | None
    suggested_monthly_inr: float | None
    suggested_tranche_inr: float | None
    max_position_pct: float
    same_sector_count_in_bucket: int | None
    review_cadence: str
    delivery_note: str
    diversification_note: str | None = None


@dataclass(frozen=True)
class PrescanSummary:
    quant_score: float | None
    quality_score: float | None
    growth_score: float | None
    strength_score: float | None
    band: str | None
    issuer_class: str | None
    route: str | None
    eligibility_verdict: str | None
    cash_conversion_status: str | None
    ocf_pat_current: float | None
    ocf_pat_3y: float | None
    data_confidence: str | None
    major_flags: tuple[str, ...]


@dataclass(frozen=True)
class NewsSummaryItem:
    date: str
    source: str
    news_type: str
    headline: str
    note: str


@dataclass(frozen=True, eq=False)
class Brief:
    ticker: TickerInfo
    price: PriceData
    technicals: Technicals
    financials: Financials | None
    shareholding: Shareholding | None
    news: NewsItems | None
    annual_report: ReportText
    missing: list[str]
    token_count: int
    confidence_ceiling: int
    generated_at: datetime
    metadata: BriefMetadata | None = None
    prescan_summary: PrescanSummary | None = None
    news_summary: tuple[NewsSummaryItem, ...] = ()
    street_consensus: StreetConsensus | None = None
    peer_snapshot: PeerSnapshot | None = None
    sector_scorecard: SectorScorecardContext | None = None
    portfolio_execution: PortfolioExecutionContext | None = None


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    failures: list[str]


@dataclass(frozen=True)
class Analysis:
    ticker: str
    run_date: date
    verdict_json: dict
    report_md: str
    costs: float
    validation: ValidationResult
    missing: list[str]
