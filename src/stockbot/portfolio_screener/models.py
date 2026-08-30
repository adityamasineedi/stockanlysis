"""Data contracts for the portfolio pre-screener.

Distinct from stockbot.models Brief/Analysis — the screener never
produces BUY/WATCH/SKIP or fair-value outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from stockbot.portfolio_screener.scoring_config import (
    CandidateBand,
    ConfidenceLevel,
    GrowthTrend,
    MoatConfidence,
    ValuationRisk,
)

HardFilterStatus = Literal["PASS", "HARD_EXCLUDE", "DATA_INSUFFICIENT", "DATA_UNAVAILABLE"]
SelectionStatus = Literal[
    "SELECTED",
    "AI_REJECTED",
    "QUANT_REJECTED",
    "HARD_EXCLUDED",
    "DIVERSIFICATION_DROPPED",
    "BELOW_THRESHOLD",
    "NOT_SENT_TO_AI",
]
ScreenStatus = Literal[
    "READY_FOR_DEEP_ANALYSIS",
    "INSUFFICIENT_HIGH_QUALITY_CANDIDATES",
    "DRY_RUN_COMPLETE",
]


@dataclass
class MetricValue:
    """A single extracted metric. None value = unavailable (never invented)."""

    name: str
    value: float | str | None
    available: bool
    missing_reason: str | None = None
    source: str | None = None


@dataclass
class StockMetrics:
    """Flattened quantitative inputs for one ticker after fetch + extract."""

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap_cr: float | None = None
    current_price_abs: float | None = None

    revenue: float | None = None
    revenue_series: list[float | None] = field(default_factory=list)
    ebitda: float | None = None
    ebitda_series: list[float | None] = field(default_factory=list)
    ebit: float | None = None
    operating_profit: float | None = None
    net_income: float | None = None
    net_income_series: list[float | None] = field(default_factory=list)
    # Same as net_income_series but with any trailing "TTM" column dropped, so it
    # aligns fiscal-year-for-fiscal-year with ocf_series_fy_only. Screener's P&L
    # table often carries a TTM column that its cash-flow table does not; summing
    # the raw "last N" tails of net_income_series/ocf_series can silently pair
    # different periods. Used only for multi-year cumulative ratios.
    net_income_series_fy_only: list[float | None] = field(default_factory=list)
    eps: float | None = None
    eps_series: list[float | None] = field(default_factory=list)

    operating_cash_flow: float | None = None
    ocf_series: list[float | None] = field(default_factory=list)
    ocf_series_fy_only: list[float | None] = field(default_factory=list)
    free_cash_flow: float | None = None
    fcf_series: list[float | None] = field(default_factory=list)

    roe: float | None = None
    roce: float | None = None
    roic: float | None = None
    roe_series: list[float | None] = field(default_factory=list)
    roce_series: list[float | None] = field(default_factory=list)

    debt: float | None = None
    debt_series: list[float | None] = field(default_factory=list)
    cash: float | None = None
    net_debt: float | None = None
    equity: float | None = None
    interest_coverage: float | None = None
    current_ratio: float | None = None
    debt_equity: float | None = None
    net_debt_ebitda: float | None = None

    operating_margin: float | None = None
    ebitda_margin: float | None = None
    operating_margin_series: list[float | None] = field(default_factory=list)

    pe: float | None = None
    forward_pe: float | None = None
    pb: float | None = None
    ev_ebitda: float | None = None
    peg: float | None = None
    price_fcf: float | None = None
    dividend_yield_pct: float | None = None

    revenue_cagr_3y: float | None = None
    revenue_cagr_5y: float | None = None
    eps_cagr_3y: float | None = None
    eps_cagr_5y: float | None = None
    ebitda_cagr_3y: float | None = None

    promoter_holding_pct: float | None = None
    pledged_promoter_holding_pct: float | None = None
    share_dilution_pct: float | None = None

    ocf_to_pat: float | None = None
    fcf_to_pat: float | None = None
    fcf_margin: float | None = None
    asset_turnover: float | None = None

    years_available: int = 0
    financials_basis: str | None = None  # consolidated | standalone (Screener)
    sector_source: str | None = None  # yfinance | override
    data_timestamp: datetime | None = None
    price_returns: list[float] | None = None  # daily returns for correlation
    missing: dict[str, str] = field(default_factory=dict)
    raw_notes: list[str] = field(default_factory=list)
    # Where a filled metric came from: "fetched" | "computed" | "yfinance"
    metric_sources: dict[str, str] = field(default_factory=dict)


@dataclass
class DataValidationResult:
    ticker: str
    data_completeness_score: float
    data_quality_score: float
    data_confidence: ConfidenceLevel
    missing_metrics: dict[str, str]
    contradictions: list[str]
    critical_ok: bool


@dataclass
class HardFilterResult:
    ticker: str
    status: HardFilterStatus
    reasons: list[str]
    valuation_risk: ValuationRisk | None = None
    human_override: bool = False


@dataclass
class RedFlag:
    severity: Literal["severe", "major", "moderate", "minor"]
    code: str
    message: str
    penalty: float


@dataclass
class ComponentScores:
    business_quality: float
    financial_strength: float
    growth: float
    cash_flow_quality: float
    capital_efficiency: float
    valuation: float
    balance_sheet: float
    earnings_quality: float
    risk: float
    growth_quality: float | None = None
    growth_trend: GrowthTrend | None = None
    valuation_risk: ValuationRisk = "MEDIUM"
    valuation_percentile: float | None = None
    valuation_confidence: ConfidenceLevel = "LOW"
    moat_confidence: MoatConfidence = "LOW"
    sector_percentile: float | None = None
    industry_percentile: float | None = None


@dataclass
class QuantScreenResult:
    ticker: str
    base_score: float
    red_flag_penalty: float
    final_quant_score: float
    components: ComponentScores
    red_flags: list[RedFlag]
    data_validation: DataValidationResult
    hard_filter: HardFilterResult
    sector: str | None
    industry: str | None
    data_timestamp: datetime | None = None
    # Price the scan saw, carried so screening records can be measured forward
    # (realized return since scan) rather than only compared against each other.
    current_price_abs: float | None = None


@dataclass
class AIRankResult:
    ticker: str
    rank: int
    ai_score: float
    confidence: ConfidenceLevel
    keep_for_deep_analysis: bool
    key_reason: str
    key_risk: str
    data_concerns: list[str]


@dataclass
class StockScreenRecord:
    """Full audit trail for one stock across the screening pipeline."""

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None

    hard_filter_status: HardFilterStatus = "PASS"
    hard_filter_reason: list[str] = field(default_factory=list)

    quant_score: float | None = None
    base_score: float | None = None
    red_flag_penalty: float = 0.0
    ai_score: float | None = None
    final_score: float | None = None
    ranking: int | None = None

    quality_score: float | None = None
    growth_score: float | None = None
    valuation_score: float | None = None
    financial_strength_score: float | None = None
    risk_score: float | None = None
    cash_flow_score: float | None = None
    capital_efficiency_score: float | None = None
    balance_sheet_score: float | None = None
    earnings_quality_score: float | None = None

    valuation_risk: ValuationRisk | None = None
    growth_trend: GrowthTrend | None = None
    data_confidence: ConfidenceLevel | None = None
    data_completeness: float | None = None
    data_quality: float | None = None

    selection_status: SelectionStatus = "QUANT_REJECTED"
    selection_reason: str = ""
    rejection_reason: str = ""
    key_risks: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)

    correlation_risk: str | None = None
    correlation_cluster: str | None = None
    candidate_band: CandidateBand | None = None

    sent_to_ai: bool = False
    ai_detail: AIRankResult | None = None

    # Entry snapshot — without these a screening record can never be scored
    # against what the stock actually did afterwards.
    price_at_scan: float | None = None
    scanned_at: datetime | None = None


@dataclass
class CostSummary:
    stocks_processed: int = 0
    ai_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_inr: float = 0.0
    estimated_deep_analysis_cost_saved_inr: float = 0.0


@dataclass
class ScreeningResult:
    universe_size: int
    hard_excluded: int
    data_insufficient: int
    quant_screened: int
    sent_to_ai: int
    final_candidates: int
    status: ScreenStatus
    stocks: list[StockScreenRecord]
    rejected: list[StockScreenRecord]
    costs: CostSummary
    screening_version: str
    weights_version: str
    prompt_version: str
    ai_model: str
    data_timestamp: datetime
    universe_timestamp: datetime
    human_table: str
    deep_analysis_tickers: list[str] = field(default_factory=list)
    deep_analysis_results: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _stock(s: StockScreenRecord) -> dict[str, Any]:
            return {
                "ticker": s.ticker,
                "rank": s.ranking,
                "quant_score": s.quant_score,
                "ai_score": s.ai_score,
                "final_score": s.final_score,
                "sector": s.sector,
                "industry": s.industry,
                "quality_score": s.quality_score,
                "growth_score": s.growth_score,
                "valuation_score": s.valuation_score,
                "financial_strength_score": s.financial_strength_score,
                "risk_score": s.risk_score,
                "data_confidence": s.data_confidence,
                "selection_reason": s.selection_reason,
                "selection_status": s.selection_status,
                "rejection_reason": s.rejection_reason,
                "key_risks": s.key_risks,
                "hard_filter_status": s.hard_filter_status,
                "hard_filter_reason": s.hard_filter_reason,
                "correlation_risk": s.correlation_risk,
                "correlation_cluster": s.correlation_cluster,
                "candidate_band": s.candidate_band,
            }

        return {
            "universe_size": self.universe_size,
            "hard_excluded": self.hard_excluded,
            "data_insufficient": self.data_insufficient,
            "quant_screened": self.quant_screened,
            "sent_to_ai": self.sent_to_ai,
            "final_candidates": self.final_candidates,
            "status": self.status,
            "stocks": [_stock(s) for s in self.stocks],
            "rejected": [_stock(s) for s in self.rejected],
            "costs": {
                "stocks_processed": self.costs.stocks_processed,
                "AI_calls": self.costs.ai_calls,
                "input_tokens": self.costs.input_tokens,
                "output_tokens": self.costs.output_tokens,
                "estimated_cost": self.costs.estimated_cost_inr,
                "estimated_deep_analysis_cost_saved_inr": (
                    self.costs.estimated_deep_analysis_cost_saved_inr
                ),
            },
            "screening_version": self.screening_version,
            "weights_version": self.weights_version,
            "prompt_version": self.prompt_version,
            "ai_model": self.ai_model,
            "data_timestamp": self.data_timestamp.isoformat(),
            "universe_timestamp": self.universe_timestamp.isoformat(),
            "deep_analysis_tickers": self.deep_analysis_tickers,
        }
