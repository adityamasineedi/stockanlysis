"""Configurable weights, thresholds, and version stamps for the
portfolio pre-screener. All tuning knobs live here — scorers must not
hard-code weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SCREENING_VERSION = "v1.0"
WEIGHTS_VERSION = "v1.0"
PROMPT_VERSION = "v1.0"

# Pre-screen AI ranking defaults. gpt-4o-mini is the cheapest solid option
# among keys this project supports (~$0.15/$0.60 vs DeepSeek Flash $0.22/$0.66
# off-peak vs Haiku $1/$5). resolve_ai_ranker() picks by available keys.
AI_PROVIDER_AUTO = "auto"
AI_RANKER_PROVIDER_DEFAULT = AI_PROVIDER_AUTO
AI_RANKER_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-v4-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}
# Legacy alias — kept so older imports still resolve.
AI_RANKER_MODEL = AI_RANKER_MODELS["openai"]

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]
ValuationRisk = Literal["LOW", "MEDIUM", "HIGH", "EXTREME"]
GrowthTrend = Literal["ACCELERATING", "STABLE", "DECELERATING", "NEGATIVE"]
MoatConfidence = Literal["LOW", "MEDIUM", "HIGH"]
CandidateBand = Literal["STRONG_CANDIDATE", "CANDIDATE", "WATCHLIST", "REMOVE"]


@dataclass(frozen=True)
class ScreeningWeights:
    business_quality: float = 20.0
    financial_strength: float = 15.0
    growth: float = 15.0
    cash_flow_quality: float = 10.0
    capital_efficiency: float = 10.0
    valuation: float = 15.0
    balance_sheet: float = 5.0
    earnings_quality: float = 5.0
    risk: float = 5.0

    def total(self) -> float:
        return (
            self.business_quality
            + self.financial_strength
            + self.growth
            + self.cash_flow_quality
            + self.capital_efficiency
            + self.valuation
            + self.balance_sheet
            + self.earnings_quality
            + self.risk
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "business_quality": self.business_quality,
            "financial_strength": self.financial_strength,
            "growth": self.growth,
            "cash_flow_quality": self.cash_flow_quality,
            "capital_efficiency": self.capital_efficiency,
            "valuation": self.valuation,
            "balance_sheet": self.balance_sheet,
            "earnings_quality": self.earnings_quality,
            "risk": self.risk,
        }


SCREENING_WEIGHTS = ScreeningWeights()


@dataclass(frozen=True)
class FinalScoreBlend:
    quant_weight: float = 0.70
    ai_weight: float = 0.30

    def validate(self) -> None:
        total = self.quant_weight + self.ai_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Final score blend must sum to 1.0, got {total}")


FINAL_SCORE_BLEND = FinalScoreBlend()


@dataclass(frozen=True)
class PortfolioConstraints:
    min_stocks: int = 10
    max_stocks: int = 18
    max_sector_weight: float = 0.30
    max_industry_weight: float = 0.20
    min_final_score: float = 60.0
    strong_candidate_min: float = 80.0
    candidate_min: float = 70.0
    watchlist_min: float = 60.0
    ai_shortlist_size: int = 25
    correlation_cluster_threshold: float = 0.85
    max_per_correlation_cluster: int = 2


PORTFOLIO_CONSTRAINTS = PortfolioConstraints()


@dataclass(frozen=True)
class RedFlagPenalties:
    severe: float = -20.0
    major: float = -10.0
    moderate: float = -5.0
    minor: float = -2.0


RED_FLAG_PENALTIES = RedFlagPenalties()


@dataclass(frozen=True)
class HardFilterThresholds:
    """Configurable hard-exclusion thresholds. Stocks failing these are
    marked HARD_EXCLUDE and cannot be AI-promoted back into the pool."""

    min_interest_coverage: float = 1.5
    max_debt_equity: float = 3.0
    max_net_debt_ebitda: float = 5.0
    persistent_negative_ocf_years: int = 3
    persistent_loss_years: int = 3
    max_promoter_pledge_pct: float = 50.0
    require_critical_metrics: tuple[str, ...] = (
        "current_price_abs",
        "revenue",
        "net_income",
        "eps",
        "operating_cash_flow",
        # ROE is preferred but may be derived from P&L+BS when Screener
        # omits the ratios row (found live: BBOX). It is gated via
        # key_trio_metrics instead of always forcing DATA_INSUFFICIENT.
    )
    # Gatekeeper key trio — DATA_INSUFFICIENT only when ≥2 of these are still
    # missing after fetched/computed fallbacks (v1.3 confidence alignment).
    key_trio_metrics: tuple[str, ...] = (
        "roe",
        "debt_equity",
        "ocf_to_pat",
    )
    allow_human_override: bool = False


HARD_FILTER_THRESHOLDS = HardFilterThresholds()


@dataclass(frozen=True)
class SectorValuationBenchmarks:
    """Median-ish sector P/E / EV-EBITDA reference points for relative
    scoring. Used only when peer percentiles cannot be computed from the
    current universe. Never fabricated into historical percentiles."""

    pe_fair: float
    pe_expensive: float
    ev_ebitda_fair: float | None = None
    pb_fair: float | None = None
    preferred_metrics: tuple[str, ...] = ("pe", "ev_ebitda")


DEFAULT_SECTOR_BENCHMARKS: dict[str, SectorValuationBenchmarks] = {
    "Technology": SectorValuationBenchmarks(28.0, 45.0, 18.0, None, ("pe", "ev_ebitda")),
    "Information Technology": SectorValuationBenchmarks(
        28.0, 45.0, 18.0, None, ("pe", "ev_ebitda")
    ),
    "Financial Services": SectorValuationBenchmarks(15.0, 25.0, None, 2.0, ("pb", "pe")),
    "Banks": SectorValuationBenchmarks(12.0, 20.0, None, 1.8, ("pb", "pe")),
    "Consumer Defensive": SectorValuationBenchmarks(35.0, 55.0, 22.0, None, ("pe", "ev_ebitda")),
    "Consumer Cyclical": SectorValuationBenchmarks(30.0, 50.0, 18.0, None, ("pe", "ev_ebitda")),
    "Healthcare": SectorValuationBenchmarks(30.0, 50.0, 20.0, None, ("pe", "ev_ebitda")),
    "Industrials": SectorValuationBenchmarks(25.0, 40.0, 14.0, None, ("pe", "ev_ebitda", "roce")),
    "Basic Materials": SectorValuationBenchmarks(18.0, 30.0, 10.0, None, ("pe", "ev_ebitda", "roce")),
    "Energy": SectorValuationBenchmarks(12.0, 22.0, 8.0, None, ("pe", "ev_ebitda")),
    "Utilities": SectorValuationBenchmarks(15.0, 25.0, 10.0, None, ("pe", "pb")),
    "Communication Services": SectorValuationBenchmarks(
        22.0, 35.0, 12.0, None, ("pe", "ev_ebitda")
    ),
    "Real Estate": SectorValuationBenchmarks(20.0, 35.0, None, 2.5, ("pb", "pe")),
    "Unknown": SectorValuationBenchmarks(22.0, 40.0, 14.0, 3.0, ("pe", "ev_ebitda")),
}


@dataclass(frozen=True)
class ScreenerRunConfig:
    weights: ScreeningWeights = field(default_factory=ScreeningWeights)
    blend: FinalScoreBlend = field(default_factory=FinalScoreBlend)
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    red_flag_penalties: RedFlagPenalties = field(default_factory=RedFlagPenalties)
    hard_filters: HardFilterThresholds = field(default_factory=HardFilterThresholds)
    sector_benchmarks: dict[str, SectorValuationBenchmarks] = field(
        default_factory=lambda: dict(DEFAULT_SECTOR_BENCHMARKS)
    )
    screening_version: str = SCREENING_VERSION
    weights_version: str = WEIGHTS_VERSION
    prompt_version: str = PROMPT_VERSION
    # "auto" | "openai" | "deepseek" | "anthropic"
    ai_provider: str = AI_RANKER_PROVIDER_DEFAULT
    ai_model: str | None = None  # None → provider default from AI_RANKER_MODELS
    dry_run: bool = False
    skip_ai: bool = False
    run_deep_analysis: bool = False
    max_deep_analyses: int | None = None


DEFAULT_RUN_CONFIG = ScreenerRunConfig()
