"""Issuer classification, cash-conversion status, and eligibility routing.

Prevents one-size-fits-all non-financial cliffs (single-year OCF/PAT, bank
D/E, utility leverage) from auto-rejecting otherwise strong names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from stockbot.portfolio_screener.models import QuantScreenResult, StockMetrics
from stockbot.portfolio_screener.score_utils import series_present

IssuerClass = Literal[
    "NON_FINANCIAL",
    "BANK",
    "NBFC_HFC",
    "INSURER",
    "UTILITY",
    "DEFENCE_EPC_PROJECT",
    "CONGLOMERATE",
    "LOSS_MAKING_GROWTH",
    "OTHER",
]

CashConversionStatus = Literal[
    "PASS",
    "WATCH",
    "ESCALATED_WATCH",
    "CRITICAL",
    "NOT_APPLICABLE",
    "NOT_APPLICABLE_WHILE_LOSS_MAKING",
    "DATA_INSUFFICIENT_FOR_TREND",
]

NextResearchAction = Literal[
    "FULL_DEEP_ANALYSIS",
    "CHEAP_WC_RECONCILIATION_FIRST",
    "SECTOR_SCORECARD_FIRST",
    "HOLDING_MONITOR",
    "DATA_RETRY",
    "NO_RESEARCH",
]

EligibilityRoute = Literal[
    "AUTO_DEEP",
    "SECTOR_SPECIFIC_REVIEW",
    "BANK_SCORECARD",
    "UTILITY_DEEP_REVIEW",
    "DEFENCE_WC_REVIEW",
    "CONGLOMERATE_SOTP_REVIEW",
    "EXCEPTION_DEEP_REVIEW",
    "LOSS_MAKING_GROWTH_FRAMEWORK",
    "HOLDING_MONITOR",
    "DATA_RETRY",
    "REJECT",
]

# Extremely weak cumulative conversion for WC-sensitive issuers (Mazdock-style).
_ESCALATED_OCF_PAT_3Y = 0.25

# Cheap WC research must classify the cash-flow gap as one of these before
# any buy/add-range analysis. Only TEMPORARY_BILLING_CYCLE with evidence unlocks
# full valuation / three-year capital-allocation ranges.
WcGapClassification = Literal[
    "TEMPORARY_BILLING_CYCLE",
    "WORKING_CAPITAL_STRESS",
    "DATA_OR_SCOPE_ERROR",
    "INCONCLUSIVE",
]

WC_RECONCILIATION_CHECKLIST: tuple[str, ...] = (
    "Verify CFO and PAT use the same period and statement scope "
    "(consolidated with consolidated, or standalone with standalone).",
    "Pull CFO, PAT, receivables, inventory, contract assets/liabilities, "
    "customer advances, and capex for each of the last 3–5 years.",
    "Explain the year-by-year cash bridge: "
    "PAT → non-cash items → working-capital changes → CFO.",
    "Compare receivables, inventory, and contract assets growth with revenue "
    "and order-book growth.",
    "Check whether customer advances and milestone payments are rising or falling.",
    "Check whether contract execution, delivery, or government/customer collection "
    "timing explains the cash-flow gap.",
    "Confirm that the order book is executable, funded, and not merely announced.",
)

WC_GAP_UNLOCKS_VALUATION: frozenset[str] = frozenset({"TEMPORARY_BILLING_CYCLE"})

# Tickers with known multi-segment / conglomerate structure (NSE).
_CONGLOMERATE_TICKERS = frozenset(
    {
        "RELIANCE",
        "TATASTEEL",  # group co. — keep narrow; RELIANCE is the critical one
        "ADANIENT",
        "ADANIPORTS",
    }
)

_BANK_SECTOR_KEYS = (
    "financial services",
    "banks",
    "bank",
)
_NBFC_KEYS = ("nbfc", "housing finance", "hfc", "consumer finance")
_INSURER_KEYS = ("insurance", "life insurance", "general insurance")
_UTILITY_KEYS = ("utilities", "utility", "power", "electric", "gas utilities")
_DEFENCE_TICKERS = frozenset({"BEL", "HAL", "BHEL", "MAZDOCK", "COCHINSHIP", "GRSE"})
_EPC_PROJECT_KEYS = ("engineering", "construction", "epc", "infrastructure")


@dataclass(frozen=True)
class CashConversionAssessment:
    status: CashConversionStatus
    ocf_pat_current: float | None
    ocf_pat_3y: float | None  # cumulative ΣOCF / ΣPAT over aligned years (not avg)
    reason: str
    ocf_pat_yearly: tuple[float | None, ...] = ()
    cfo_3y_sum: float | None = None
    pat_3y_sum: float | None = None
    years_used: int = 0
    interpretation: str = ""
    ocf_current: float | None = None  # absolute CFO (₹ Cr) for loss-making display


def fmt_ratio(value: float | None, decimals: int = 2) -> str:
    """User-facing ratio text — never expose raw float noise."""
    if value is None:
        return "null"
    return f"{value:.{decimals}f}"


def fmt_cr(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "null"
    return f"{value:.{decimals}f}"


_ISSUER_CASHFLOW_CONTEXT: dict[IssuerClass, str] = {
    "DEFENCE_EPC_PROJECT": (
        "defence/project business with possible milestone billing and working-capital timing"
    ),
    "UTILITY": (
        "capital-intensive utility with regulated/project cash-flow cycles"
    ),
    "CONGLOMERATE": "diversified group with segment-specific cash-flow cycles",
    "NON_FINANCIAL": "operating business",
    "OTHER": "operating business",
    "BANK": "bank",
    "NBFC_HFC": "NBFC/HFC",
    "INSURER": "insurer",
    "LOSS_MAKING_GROWTH": "loss-making growth business",
}


@dataclass(frozen=True)
class RoutingDecision:
    issuer_class: IssuerClass
    cash_conversion: CashConversionAssessment
    route: EligibilityRoute
    eligibility: str  # EligibilityVerdict value
    suitable_for_deep_analysis: bool
    key_reason: str
    key_risk: str
    quality_override: bool = False
    next_action: NextResearchAction = "NO_RESEARCH"


def classify_issuer(metrics: StockMetrics) -> IssuerClass:
    ticker = (metrics.ticker or "").upper()
    sector = (metrics.sector or "").lower()
    industry = (metrics.industry or "").lower()
    blob = f"{sector} {industry}"

    if any(k in blob for k in _INSURER_KEYS):
        return "INSURER"
    if any(k in blob for k in _NBFC_KEYS):
        return "NBFC_HFC"
    if any(k in blob for k in _BANK_SECTOR_KEYS) or "bank" in industry:
        return "BANK"
    if ticker in _CONGLOMERATE_TICKERS or "conglomerate" in blob:
        return "CONGLOMERATE"
    if any(k in blob for k in _UTILITY_KEYS):
        return "UTILITY"
    if (
        ticker in _DEFENCE_TICKERS
        or "defense" in blob
        or "defence" in blob
        or ("aerospace" in blob and "equipment" in blob)
    ):
        return "DEFENCE_EPC_PROJECT"
    # Project / EPC businesses share WC-timing OCF patterns with defence suppliers
    if any(k in blob for k in _EPC_PROJECT_KEYS):
        return "DEFENCE_EPC_PROJECT"

    # Persistent losses → loss-making growth bucket (Swiggy-style)
    pat = series_present(metrics.net_income_series)
    if len(pat) >= 3 and all(v < 0 for v in pat[-3:]):
        return "LOSS_MAKING_GROWTH"

    if sector in {"", "unknown"} and industry in {"", "unknown"}:
        return "OTHER"
    return "NON_FINANCIAL"


def _ratio(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _aligned_ocf_pat_panel(
    metrics: StockMetrics,
) -> tuple[
    list[float | None],
    float | None,
    float | None,
    float | None,
    int,
]:
    """Build yearly OCF/PAT and cumulative 3y ratio from aligned series tails.

    Cumulative 3y = ΣOCF / ΣPAT over the same N fiscal years (not an average of
    annual ratios). Requires matching series lengths from the end.
    """
    ocf_raw = list(metrics.ocf_series or [])
    pat_raw = list(metrics.net_income_series or [])
    if not ocf_raw or not pat_raw:
        return [], None, None, None, 0

    n = min(3, len(ocf_raw), len(pat_raw))
    ocf_tail = ocf_raw[-n:]
    pat_tail = pat_raw[-n:]
    yearly: list[float | None] = []
    for o, p in zip(ocf_tail, pat_tail, strict=True):
        if o is None or p is None or p == 0:
            yearly.append(None)
        else:
            yearly.append(o / p)

    # Only sum years where both values are present
    pairs = [
        (o, p)
        for o, p in zip(ocf_tail, pat_tail, strict=True)
        if o is not None and p is not None
    ]
    if len(pairs) < 2:
        return yearly, None, None, None, len(pairs)

    sum_ocf = sum(o for o, _ in pairs)
    sum_pat = sum(p for _, p in pairs)
    years_used = len(pairs)
    if sum_pat > 0:
        cum = sum_ocf / sum_pat
    elif sum_pat < 0 and sum_ocf < 0:
        cum = sum_ocf / sum_pat
    else:
        cum = None
    return yearly, cum, sum_ocf, sum_pat, years_used


def assess_cash_conversion(
    metrics: StockMetrics,
    issuer_class: IssuerClass,
) -> CashConversionAssessment:
    if issuer_class in {"BANK", "NBFC_HFC", "INSURER"}:
        return CashConversionAssessment(
            status="NOT_APPLICABLE",
            ocf_pat_current=None,
            ocf_pat_3y=None,
            reason="OCF/PAT not a primary quality gate for financials",
            interpretation="Use bank scorecard metrics (NIM, GNPA, PCR, CAR, P/B).",
        )

    yearly, ocf_pat_3y, cfo_sum, pat_sum, years_used = _aligned_ocf_pat_panel(metrics)
    ocf_current = metrics.operating_cash_flow
    if ocf_current is None and metrics.ocf_series:
        present = [v for v in metrics.ocf_series if v is not None]
        ocf_current = present[-1] if present else None

    # Loss-making: OCF/PAT is meaningless when PAT (or cumulative PAT) is negative.
    if issuer_class == "LOSS_MAKING_GROWTH":
        current_ratio = metrics.ocf_to_pat
        return CashConversionAssessment(
            status="NOT_APPLICABLE_WHILE_LOSS_MAKING",
            ocf_pat_current=current_ratio,
            ocf_pat_3y=ocf_pat_3y if years_used >= 2 else None,
            reason=(
                "OCF/PAT not classified while loss-making — negative PAT makes "
                "the ratio misleading"
            ),
            ocf_pat_yearly=tuple(yearly),
            cfo_3y_sum=cfo_sum if years_used >= 2 else None,
            pat_3y_sum=pat_sum if years_used >= 2 else None,
            years_used=years_used,
            ocf_current=ocf_current,
            interpretation=(
                "OCF/PAT is not meaningful while PAT is negative (a ratio of two "
                "negatives can look 'good' while cash is still burning). Assess "
                "absolute OCF/FCF burn, cash runway, contribution-margin trajectory, "
                "dilution/funding risk, and path to profitability instead."
            ),
        )

    current = metrics.ocf_to_pat
    if current is None and metrics.operating_cash_flow is not None and metrics.net_income:
        current = _ratio(metrics.operating_cash_flow, metrics.net_income)

    # Prefer last yearly ratio if current missing
    if current is None and yearly:
        current = next((y for y in reversed(yearly) if y is not None), None)

    # Also disable profitable OCF/PAT gates when latest or cumulative PAT is negative
    latest_pat = metrics.net_income
    if latest_pat is None and metrics.net_income_series:
        pats = [v for v in metrics.net_income_series if v is not None]
        latest_pat = pats[-1] if pats else None
    if (latest_pat is not None and latest_pat < 0) or (
        pat_sum is not None and pat_sum < 0 and years_used >= 2
    ):
        return CashConversionAssessment(
            status="NOT_APPLICABLE_WHILE_LOSS_MAKING",
            ocf_pat_current=current,
            ocf_pat_3y=ocf_pat_3y if years_used >= 2 else None,
            reason="OCF/PAT not classified while PAT (or cumulative PAT) is negative",
            ocf_pat_yearly=tuple(yearly),
            cfo_3y_sum=cfo_sum if years_used >= 2 else None,
            pat_3y_sum=pat_sum if years_used >= 2 else None,
            years_used=years_used,
            ocf_current=ocf_current,
            interpretation=(
                "OCF/PAT thresholds apply only to profitable years. Use absolute "
                "cash generation/burn instead."
            ),
        )

    wc_sensitive = issuer_class in {"DEFENCE_EPC_PROJECT", "UTILITY"}
    ctx = _ISSUER_CASHFLOW_CONTEXT.get(issuer_class, "operating business")

    def _pack(
        status: CashConversionStatus,
        reason: str,
        interpretation: str,
    ) -> CashConversionAssessment:
        return CashConversionAssessment(
            status=status,
            ocf_pat_current=current,
            ocf_pat_3y=ocf_pat_3y if years_used >= 2 else None,
            reason=reason,
            ocf_pat_yearly=tuple(yearly),
            cfo_3y_sum=cfo_sum if years_used >= 2 else None,
            pat_3y_sum=pat_sum if years_used >= 2 else None,
            years_used=years_used,
            interpretation=interpretation,
            ocf_current=ocf_current,
        )

    if current is None and (ocf_pat_3y is None or years_used < 2):
        return _pack(
            "DATA_INSUFFICIENT_FOR_TREND",
            "OCF/PAT trend unavailable — insufficient aligned CFO/PAT years",
            "Do not label a 3y cumulative ratio without aligned fiscal years.",
        )

    # WC-sensitive: never CRITICAL; escalate when 3y cumulative is extremely weak
    if wc_sensitive and (
        (current is not None and current < 0.50)
        or (ocf_pat_3y is not None and ocf_pat_3y < 0.50)
    ):
        if ocf_pat_3y is not None and years_used >= 2 and ocf_pat_3y < _ESCALATED_OCF_PAT_3Y:
            return _pack(
                "ESCALATED_WATCH",
                (
                    f"Current OCF/PAT {fmt_ratio(current)}; 3y cumulative OCF/PAT "
                    f"{fmt_ratio(ocf_pat_3y)} (ΣCFO {fmt_cr(cfo_sum)} / ΣPAT "
                    f"{fmt_cr(pat_sum)} over {years_used}y) < "
                    f"{fmt_ratio(_ESCALATED_OCF_PAT_3Y)} — extreme for {issuer_class}"
                ),
                (
                    "Reported cash conversion is extremely weak. This may reflect "
                    "milestone billing and project working-capital timing, but it is "
                    "too weak to assume without a year-by-year CFO-to-PAT reconciliation "
                    "(advances, inventory, receivables, contract assets, statement scope)."
                ),
            )
        if ocf_pat_3y is not None and ocf_pat_3y >= 0.80:
            interp = (
                f"Current OCF/PAT {fmt_ratio(current)} weak but 3y cumulative "
                f"{fmt_ratio(ocf_pat_3y)} ≥ 0.80 — may be project-cycle timing; "
                "still reconcile WC bridge before assuming temporary."
            )
        else:
            interp = (
                f"Below normal long-term threshold (current {fmt_ratio(current)}, "
                f"3y cumulative {fmt_ratio(ocf_pat_3y)}). Because this is a {ctx}, "
                "verify whether reported weakness is temporary billing-cycle timing "
                "or persistent WC stress before any buy/add range."
            )
        return _pack(
            "WATCH",
            (
                f"Current OCF/PAT {fmt_ratio(current)}; 3y cumulative OCF/PAT "
                f"{fmt_ratio(ocf_pat_3y)} for {issuer_class} — WC / billing-cycle "
                "review (not auto hard-fail)"
            ),
            interp,
        )

    # Non-WC: CRITICAL only when both current and 3y are weak
    if (
        current is not None
        and current < 0.50
        and ocf_pat_3y is not None
        and years_used >= 2
        and ocf_pat_3y < 0.50
    ):
        return _pack(
            "CRITICAL",
            (
                f"OCF/PAT current {fmt_ratio(current)} and 3y cumulative "
                f"{fmt_ratio(ocf_pat_3y)} both < 0.50"
            ),
            "Persistent weak cash conversion — profits not translating into OCF.",
        )

    if current is not None and current < 0.50:
        return _pack(
            "WATCH",
            (
                f"OCF/PAT current {fmt_ratio(current)} < 0.50 — monitor; "
                f"3y cumulative={fmt_ratio(ocf_pat_3y)}"
            ),
            "Single-year weakness; confirm multi-year trend before sizing.",
        )

    return _pack(
        "PASS",
        "Cash conversion acceptable",
        "Current and/or cumulative OCF/PAT within normal long-term bands.",
    )


def decide_next_research_action(
    *,
    issuer: IssuerClass,
    cash: CashConversionAssessment,
    eligibility: str,
) -> NextResearchAction:
    if eligibility == "DATA_UNAVAILABLE_RETRY":
        return "DATA_RETRY"
    if eligibility in {"HOLDING_MONITOR_ONLY", "NOT_SUITABLE_FOR_3Y_RESEARCH"}:
        return "HOLDING_MONITOR" if eligibility == "HOLDING_MONITOR_ONLY" else "NO_RESEARCH"
    if issuer in {"BANK", "NBFC_HFC", "INSURER"}:
        return "SECTOR_SCORECARD_FIRST"
    if issuer == "DEFENCE_EPC_PROJECT":
        if cash.status == "ESCALATED_WATCH":
            return "CHEAP_WC_RECONCILIATION_FIRST"
        if (
            cash.ocf_pat_3y is not None
            and cash.years_used >= 2
            and cash.ocf_pat_3y < _ESCALATED_OCF_PAT_3Y
        ):
            return "CHEAP_WC_RECONCILIATION_FIRST"
        if cash.status in {"WATCH", "DATA_INSUFFICIENT_FOR_TREND"}:
            # BEL-style: full deep allowed but valuation ranges gated on WC section
            return "FULL_DEEP_ANALYSIS"
        return "FULL_DEEP_ANALYSIS"
    if eligibility in {"AUTO_DEEP_ANALYSIS", "SECTOR_SPECIFIC_REVIEW"}:
        return "FULL_DEEP_ANALYSIS"
    return "NO_RESEARCH"


def fundamentals_fetch_failed(metrics: StockMetrics) -> bool:
    """True only when the fetch layer failed — not when ratios are merely incomplete."""
    markers = (
        "fundamentals fetch failed",
        "fundamentals:",
        "fetch failed",
        "yfinance error",
        "http error",
        "timeout",
    )
    blob = " ".join(str(v).lower() for v in metrics.missing.values())
    if any(m in blob for m in markers):
        return True
    return False


def quality_override_applies(quant: QuantScreenResult) -> bool:
    c = quant.components
    # Only governance (pledge) severe blocks override — OCF_PAT_WATCH is major/minor
    gov_severe = any(
        f.severity == "severe" and f.code.startswith("PLEDGE") for f in quant.red_flags
    )
    if gov_severe:
        return False
    # Round to match Telegram Quality/Growth/Strength display so 74.95 → 75 qualifies
    return (
        round(c.business_quality) >= 75
        and round(c.growth) >= 65
        and round(c.financial_strength) >= 70
    )


def decide_eligibility_route(
    metrics: StockMetrics,
    quant: QuantScreenResult,
) -> RoutingDecision:
    issuer = classify_issuer(metrics)
    cash = assess_cash_conversion(metrics, issuer)
    hard = quant.hard_filter.status
    score = quant.final_quant_score
    c = quant.components
    override = quality_override_applies(quant)

    def _fin(decision: RoutingDecision) -> RoutingDecision:
        return replace(
            decision,
            next_action=decide_next_research_action(
                issuer=decision.issuer_class,
                cash=decision.cash_conversion,
                eligibility=decision.eligibility,
            ),
        )

    if fundamentals_fetch_failed(metrics) or hard == "DATA_UNAVAILABLE":
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="DATA_RETRY",
            eligibility="DATA_UNAVAILABLE_RETRY",
            suitable_for_deep_analysis=False,
            key_reason=(
                "Fundamentals fetch failed or empty — no 3y research conclusion"
            ),
            key_risk="Retry with NSE symbol / Screener fallback; do not treat as weak quality",
            quality_override=False,
        ))

    if issuer in {"BANK", "NBFC_HFC", "INSURER"}:
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="BANK_SCORECARD",
            eligibility="SECTOR_SPECIFIC_REVIEW",
            suitable_for_deep_analysis=True,
            key_reason=(
                f"Issuer class {issuer}: banking/NBFC scorecard required "
                f"(generic quant {score:.1f} is not decisive) — NIM, GNPA, PCR, CAR, P/B"
            ),
            key_risk="Do not use OCF/PAT, D/E, or interest cover as primary rejection criteria",
            quality_override=False,
        ))

    if hard == "HARD_EXCLUDE":
        if issuer == "LOSS_MAKING_GROWTH":
            return _fin(RoutingDecision(
                issuer_class=issuer,
                cash_conversion=cash,
                route="LOSS_MAKING_GROWTH_FRAMEWORK",
                eligibility="NOT_SUITABLE_FOR_3Y_RESEARCH",
                suitable_for_deep_analysis=False,
                key_reason="; ".join(quant.hard_filter.reasons) or "HARD_EXCLUDE",
                key_risk=(
                    "Outside profitable-compounder 3y screen — use loss-making growth "
                    "framework if intentional. If already held: not an automatic sell. "
                    "OCF/PAT is not a meaningful pass/fail while PAT is negative."
                ),
                quality_override=False,
            ))
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="REJECT",
            eligibility="NOT_SUITABLE_FOR_3Y_RESEARCH",
            suitable_for_deep_analysis=False,
            key_reason="; ".join(quant.hard_filter.reasons) or "HARD_EXCLUDE",
            key_risk=(
                "Hard exclusion after applicable filters. If already held: not an "
                "automatic sell — review thesis before new capital or research spend."
            ),
            quality_override=False,
        ))

    if hard == "DATA_INSUFFICIENT":
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="DATA_RETRY",
            eligibility="DATA_UNAVAILABLE_RETRY",
            suitable_for_deep_analysis=False,
            key_reason="; ".join(quant.hard_filter.reasons) or "critical metrics missing",
            key_risk="Incomplete data — retry before concluding 3y research eligibility",
            quality_override=False,
        ))

    # --- Soft sector floors → SECTOR_SPECIFIC_REVIEW (enter research with sector lens) ---
    if issuer == "UTILITY":
        de_ok = metrics.debt_equity is None or metrics.debt_equity < 2.5
        ic_ok = metrics.interest_coverage is None or metrics.interest_coverage >= 2.0
        cash_ok = cash.status in {"PASS", "WATCH", "ESCALATED_WATCH", "NOT_APPLICABLE"}
        if de_ok and ic_ok and cash_ok:
            return _fin(RoutingDecision(
                issuer_class=issuer,
                cash_conversion=cash,
                route="UTILITY_DEEP_REVIEW",
                eligibility="SECTOR_SPECIFIC_REVIEW",
                suitable_for_deep_analysis=True,
                key_reason=(
                    f"Utility lens required: quant {score:.1f} — regulated cash flows, "
                    "debt maturity, project capex, ROCE (not generic leverage cliffs)"
                ),
                key_risk="Review regulated vs merchant mix, debt maturity, incremental ROCE",
                quality_override=False,
            ))

    if issuer == "CONGLOMERATE":
        cash_ok = cash.status in {"PASS", "WATCH", "ESCALATED_WATCH", "NOT_APPLICABLE"}
        lev_ok = metrics.debt_equity is None or metrics.debt_equity < 1.5
        if cash_ok and lev_ok:
            return _fin(RoutingDecision(
                issuer_class=issuer,
                cash_conversion=cash,
                route="CONGLOMERATE_SOTP_REVIEW",
                eligibility="SECTOR_SPECIFIC_REVIEW",
                suitable_for_deep_analysis=True,
                key_reason=(
                    f"Conglomerate/SOTP lens required: aggregate quant {score:.1f} "
                    "is not decisive"
                ),
                key_risk="Segment quality, net debt, cash generation — not aggregate P/E alone",
                quality_override=False,
            ))

    if issuer == "DEFENCE_EPC_PROJECT" and cash.status in {"WATCH", "ESCALATED_WATCH"} and override:
        cum = (
            f"{cash.ocf_pat_3y:.2f}"
            if cash.ocf_pat_3y is not None
            else "n/a"
        )
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="DEFENCE_WC_REVIEW",
            eligibility="SECTOR_SPECIFIC_REVIEW",
            suitable_for_deep_analysis=True,
            key_reason=(
                "Strong quality/growth/strength conflict with low generic score."
            ),
            key_risk=(
                f"Three-year CFO/PAT is {cum}; working-capital explanation is required."
            ),
            quality_override=True,
        ))

    if score < 60 and override and cash.status != "CRITICAL":
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="EXCEPTION_DEEP_REVIEW",
            eligibility="SECTOR_SPECIFIC_REVIEW",
            suitable_for_deep_analysis=True,
            key_reason=(
                f"Quality override: strong Q/G/S vs weak composite {score:.1f} — "
                "sector/exception deep review for 3y research"
            ),
            key_risk=(cash.interpretation or cash.reason) if cash.status in {"WATCH", "ESCALATED_WATCH"} else "Reconcile score conflict before sizing",
            quality_override=True,
        ))

    if score >= 70 and cash.status != "CRITICAL":
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="AUTO_DEEP",
            eligibility="AUTO_DEEP_ANALYSIS",
            suitable_for_deep_analysis=True,
            key_reason=f"Quant {score:.1f}; cash conversion {cash.status} — eligible for 3y deep research",
            key_risk=(cash.interpretation or cash.reason) if cash.status in {"WATCH", "ESCALATED_WATCH"} else "Standard deep-analysis capacity controls",
            quality_override=False,
        ))

    if score >= 55 and cash.status != "CRITICAL":
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="SECTOR_SPECIFIC_REVIEW",
            eligibility="SECTOR_SPECIFIC_REVIEW",
            suitable_for_deep_analysis=True,
            key_reason=f"Quant {score:.1f} — targeted sector/thesis review before auto deep spend",
            key_risk="Lower-cost focused review; not an automatic reject for 3y research",
            quality_override=False,
        ))

    # Weak for fresh research/capital — monitor if held; not a sell instruction
    if cash.status == "CRITICAL" and not override:
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="HOLDING_MONITOR",
            eligibility="HOLDING_MONITOR_ONLY",
            suitable_for_deep_analysis=False,
            key_reason=(
                f"quant_score {score:.2f}; {cash.reason} — not auto-eligible for "
                "fresh 3y research spend or new capital"
            ),
            key_risk=(
                "If already held: monitor only — NOT a sell instruction. "
                "Review thesis before adding capital. If not held: prefer "
                "NOT_SUITABLE_FOR_3Y_RESEARCH until cash conversion improves."
            ),
            quality_override=False,
        ))

    if score < 55 and not override:
        return _fin(RoutingDecision(
            issuer_class=issuer,
            cash_conversion=cash,
            route="HOLDING_MONITOR",
            eligibility="HOLDING_MONITOR_ONLY",
            suitable_for_deep_analysis=False,
            key_reason=(
                f"quant_score {score:.2f} below 3y research floor; cash {cash.status}"
            ),
            key_risk=(
                "If already held: not an automatic sell — thesis review before new "
                "capital. Fresh research spend not auto-approved."
            ),
            quality_override=False,
        ))

    return _fin(RoutingDecision(
        issuer_class=issuer,
        cash_conversion=cash,
        route="SECTOR_SPECIFIC_REVIEW",
        eligibility="SECTOR_SPECIFIC_REVIEW",
        suitable_for_deep_analysis=True,
        key_reason=f"Borderline case: quant {score:.1f}, cash {cash.status}, class {issuer}",
        key_risk="Human judgment / targeted 3y research review",
        quality_override=override,
    ))
