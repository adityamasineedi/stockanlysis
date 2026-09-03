"""Single-ticker 3-year portfolio research eligibility check.

Answers: "Should this NSE stock enter the expensive 3y deep-research workflow?"
— not BUY/WATCH/SKIP, fair value, average-down, or profit-booking.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from anthropic import Anthropic

from stockbot.config import PROMPTS_DIR, settings
from stockbot.costs import log_call
from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
from stockbot.models import AmbiguousMatch, TickerInfo
from stockbot.portfolio_screener.ai_ranker import (
    DEEPSEEK_API_BASE,
    OPENAI_API_BASE,
    resolve_ai_ranker,
)
from stockbot.portfolio_screener.cost_tracker import ScreenerCostTracker
from stockbot.portfolio_screener.data_loader import fetch_universe_metrics
from stockbot.portfolio_screener.issuer_routing import (
    WC_RECONCILIATION_CHECKLIST,
    decide_eligibility_route,
    fundamentals_fetch_failed,
)
from stockbot.portfolio_screener.metrics import count_derived_key_ratios
from stockbot.portfolio_screener.outcome_log import (
    classify_reject,
    format_computed_metric_warnings,
    log_prescan_outcome,
)
from stockbot.portfolio_screener.portfolio_selector import (
    candidate_band,
    combine_scores,
)
from stockbot.portfolio_screener.prescan_display import (
    ENTRY_HEADLINES,
    ISSUER_LABELS,
    format_action_lines,
    format_metric_lines,
    format_qgs_compact,
    synthesize_why,
    telegram_code_line,
    telegram_small_block,
)
from stockbot.portfolio_screener.quant_engine import compute_quant_score
from stockbot.portfolio_screener.red_flags import governance_notes
from stockbot.portfolio_screener.scoring_config import ScreenerRunConfig

logger = logging.getLogger(__name__)

EligibilityVerdict = Literal[
    "AUTO_DEEP_ANALYSIS",
    "SECTOR_SPECIFIC_REVIEW",
    "HOLDING_MONITOR_ONLY",
    "DATA_UNAVAILABLE_RETRY",
    "NOT_SUITABLE_FOR_3Y_RESEARCH",
    "NOT_FOUND",
    "AMBIGUOUS",
]

_LEGACY_VERDICT_MAP: dict[str, EligibilityVerdict] = {
    "SUITABLE_FOR_DEEP_ANALYSIS": "AUTO_DEEP_ANALYSIS",
    "REVIEW_EXCEPTION": "SECTOR_SPECIFIC_REVIEW",
    "MARGINAL": "SECTOR_SPECIFIC_REVIEW",
    "MODEL_NOT_APPLICABLE": "SECTOR_SPECIFIC_REVIEW",
    "DATA_UNAVAILABLE": "DATA_UNAVAILABLE_RETRY",
    "NOT_SUITABLE": "NOT_SUITABLE_FOR_3Y_RESEARCH",
}

PROMPT_PATH = PROMPTS_DIR / "portfolio-screener-eligibility-v1.md"

_DEFAULT_ELIGIBILITY_SYSTEM = """You are a gatekeeper for a three-year NSE portfolio research workflow.
Prefer portfolio-screener-eligibility-v1.md when present.

Do NOT output BUY/WATCH/SKIP, fair value, average-down, or profit-booking.
Eligibility must be one of:
AUTO_DEEP_ANALYSIS | SECTOR_SPECIFIC_REVIEW | HOLDING_MONITOR_ONLY |
DATA_UNAVAILABLE_RETRY | NOT_SUITABLE_FOR_3Y_RESEARCH

suitable_for_deep_analysis is true only for AUTO_DEEP_ANALYSIS and SECTOR_SPECIFIC_REVIEW.
NOT_SUITABLE_FOR_3Y_RESEARCH / HOLDING_MONITOR_ONLY are not sell instructions.
Return ONLY JSON.
"""


def load_eligibility_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    logger.warning("Eligibility prompt missing at %s — using inline fallback", PROMPT_PATH)
    return _DEFAULT_ELIGIBILITY_SYSTEM


def _normalize_verdict(raw: str) -> EligibilityVerdict | None:
    key = raw.strip().upper()
    valid = {
        "AUTO_DEEP_ANALYSIS",
        "SECTOR_SPECIFIC_REVIEW",
        "HOLDING_MONITOR_ONLY",
        "DATA_UNAVAILABLE_RETRY",
        "NOT_SUITABLE_FOR_3Y_RESEARCH",
        "NOT_FOUND",
        "AMBIGUOUS",
    }
    if key in valid:
        return key  # type: ignore[return-value]
    return _LEGACY_VERDICT_MAP.get(key)


@dataclass
class EligibilityResult:
    query: str
    ticker: str | None
    company_name: str | None
    verdict: EligibilityVerdict
    suitable_for_deep_analysis: bool
    quant_score: float | None = None
    ai_score: float | None = None
    final_score: float | None = None
    candidate_band: str | None = None
    # Price at scan time — paired with checked_at, this is what makes a prescan
    # measurable forward (realized return since the scan) instead of a score
    # with no outcome attached.
    price_at_scan: float | None = None
    hard_filter_status: str | None = None
    hard_filter_reasons: list[str] = field(default_factory=list)
    sector: str | None = None
    issuer_class: str | None = None
    eligibility_route: str | None = None
    cash_conversion_status: str | None = None
    cash_conversion_interpretation: str | None = None
    ocf_pat_current: float | None = None
    ocf_pat_3y_cumulative: float | None = None
    ocf_current_abs: float | None = None
    cfo_3y_sum_abs: float | None = None
    pat_3y_sum_abs: float | None = None
    debt_equity: float | None = None
    interest_coverage: float | None = None
    net_debt_ebitda: float | None = None
    roe: float | None = None
    next_research_action: str | None = None
    quality_override: bool = False
    quality_score: float | None = None
    growth_score: float | None = None
    valuation_score: float | None = None
    financial_strength_score: float | None = None
    risk_score: float | None = None
    data_confidence: str | None = None
    data_completeness: float | None = None
    financials_basis: str | None = None
    financials_source: str | None = None
    sector_source: str | None = None
    derived_metric_count: int = 0
    key_reason: str = ""
    key_risk: str = ""
    data_concerns: list[str] = field(default_factory=list)
    computed_metric_warnings: list[str] = field(default_factory=list)
    recheck_note: str | None = None
    ai_provider: str | None = None
    ai_model: str | None = None
    cost_inr: float = 0.0
    ambiguous_candidates: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checked_at"] = self.checked_at.isoformat()
        return payload

    def telegram_html(self) -> str:
        if self.verdict == "NOT_FOUND":
            return f"❓ Could not resolve <b>{_esc(self.query)}</b>."
        if self.verdict == "AMBIGUOUS":
            opts = ", ".join(_esc(c) for c in self.ambiguous_candidates[:5])
            return (
                f"❓ <b>{_esc(self.query)}</b> is ambiguous. Try one of: {opts}"
            )

        if self.next_research_action == "CHEAP_WC_RECONCILIATION_FIRST":
            return self._telegram_html_cheap_wc()

        return self._telegram_html_compact()

    def _telegram_html_compact(self) -> str:
        """Dense beginner card — score, pillars, key ratios, gate, actions."""
        name = _esc(self.ticker or self.query)
        if self.company_name:
            name = f"{name} — {_esc(self.company_name)}"

        entry_icon, entry_label = ENTRY_HEADLINES.get(
            self.verdict, ("📋", self.verdict.replace("_", " "))
        )
        header = [
            f"👀 <b>{name}</b>",
            f"{entry_icon} <b>{entry_label}</b>",
        ]
        if self.final_score is not None:
            header.append(f"<b>Score: {self.final_score:.1f}/100</b>")

        body: list[str] = []
        qgs = format_qgs_compact(
            quality=self.quality_score,
            growth=self.growth_score,
            strength=self.financial_strength_score,
        )
        if qgs:
            body.append(qgs)

        metrics = format_metric_lines(
            debt_equity=self.debt_equity,
            roe=self.roe,
            ocf_pat=self.ocf_pat_current,
            net_debt_ebitda=self.net_debt_ebitda,
            interest_coverage=self.interest_coverage,
        )
        if metrics:
            body.extend(metrics)

        why = synthesize_why(
            key_reason=self.key_reason,
            quality=self.quality_score,
            growth=self.growth_score,
            strength=self.financial_strength_score,
        )
        # Prefer a short sector/cash override when the routing meaning is richer.
        if (
            self.next_research_action == "SECTOR_SCORECARD_FIRST"
            and self.issuer_class
        ):
            why = (
                ISSUER_LABELS.get(self.issuer_class, self.issuer_class)
                + " — use a sector scorecard, not generic cash ratios."
            )
        elif (
            self.cash_conversion_status == "WATCH"
            and self.issuer_class in {"DEFENCE_EPC_PROJECT", "EPC_PROJECT_BUSINESS"}
            and self.next_research_action == "FULL_DEEP_ANALYSIS"
        ):
            why = (
                "Strong quality signals, but reported cash conversion needs "
                "explanation (common in defence / project companies)."
            )
        body.extend(["", "⚠️ WHY?", _esc(why), ""])
        body.extend(
            format_action_lines(
                verdict=self.verdict,
                suitable_for_deep_analysis=self.suitable_for_deep_analysis,
            )
        )

        if self.hard_filter_status and self.hard_filter_status != "PASS":
            reason = (
                "; ".join(self.hard_filter_reasons[:2]) if self.hard_filter_reasons else ""
            )
            hard_plain = {
                "HARD_EXCLUDE": "Failed a hard safety filter",
                "DATA_UNAVAILABLE": "Required data missing",
                "DATA_INSUFFICIENT": "Not enough history",
            }.get(self.hard_filter_status, self.hard_filter_status)
            body.append(
                f"⛔ {_esc(hard_plain)}"
                + (f" — {_esc(reason)}" if reason else "")
            )

        if self.derived_metric_count >= 3 or self.computed_metric_warnings:
            body.extend(["", "⚠️ Check calculated ratios before decisions."])
        if self.financials_basis:
            body.append(
                f"<i>{_esc(self.financials_basis)} · "
                f"{_esc(_financials_source_label(self.financials_source))}</i>"
            )
        if self.recheck_note:
            body.append(f"📅 {_esc(self.recheck_note)}")
        body.append("ℹ️ <i>Pre-scan only.</i>")

        small = telegram_small_block(body)
        return "\n".join(header + (["", small] if small else []))

    def _why_routed_blocked(self) -> tuple[str | None, str | None]:
        """Short auditable lines instead of one truncated Why paragraph."""
        if self.next_research_action == "CHEAP_WC_RECONCILIATION_FIRST":
            routed = self.key_reason or (
                "Strong quality/growth/strength conflict with low generic score."
            )
            if self.ocf_pat_3y_cumulative is not None:
                blocked = (
                    f"Three-year CFO/PAT is {self.ocf_pat_3y_cumulative:.2f}; "
                    "working-capital explanation is required."
                )
            else:
                blocked = self.key_risk or (
                    "Reported cash conversion is extremely weak; "
                    "working-capital explanation is required."
                )
            return routed, blocked

        if self.quality_override and self.final_score is not None and self.final_score < 60:
            routed = self.key_reason or (
                "Strong quality/growth/strength conflict with low generic score."
            )
            blocked = self.key_risk or None
            return routed, blocked

        return None, None

    def _telegram_html_cheap_wc(self) -> str:
        """Escalated WC card — compact gate + checklist, no premature conclusion."""
        name = _esc(self.ticker or self.query)
        if self.company_name:
            name = f"{name} — {_esc(self.company_name)}"

        header = [
            f"👀 <b>{name}</b>",
            "🧡 <b>CASH / WC CHECK FIRST</b>",
        ]
        if self.final_score is not None:
            header.append(f"<b>Score: {self.final_score:.1f}/100</b>")

        body: list[str] = []
        qgs = format_qgs_compact(
            quality=self.quality_score,
            growth=self.growth_score,
            strength=self.financial_strength_score,
        )
        if qgs:
            body.append(qgs)

        metrics = format_metric_lines(
            debt_equity=self.debt_equity,
            roe=self.roe,
            ocf_pat=self.ocf_pat_current,
            net_debt_ebitda=self.net_debt_ebitda,
            interest_coverage=self.interest_coverage,
        )
        if metrics:
            body.extend(metrics)

        if self.ocf_pat_3y_cumulative is not None:
            body.append(
                telegram_code_line(f"💧 3Y OCF/PAT {self.ocf_pat_3y_cumulative:.2f}×")
            )
        if self.cfo_3y_sum_abs is not None or self.pat_3y_sum_abs is not None:
            cfo = (
                f"₹{self.cfo_3y_sum_abs:.0f} Cr"
                if self.cfo_3y_sum_abs is not None
                else "?"
            )
            pat = (
                f"₹{self.pat_3y_sum_abs:.0f} Cr"
                if self.pat_3y_sum_abs is not None
                else "?"
            )
            body.append(telegram_code_line(f"💧 3Y totals: CFO {cfo} / PAT {pat}"))

        why_routed, why_blocked = self._why_routed_blocked()
        body.extend(
            [
                "",
                "⚠️ WHY?",
                _esc(
                    "Strong quality/growth/strength, but reported cash conversion "
                    "is extremely weak — reconcile working capital before research."
                ),
            ]
        )
        if why_routed:
            body.append(f"📌 Why this route: {_esc(why_routed)}")
        if why_blocked:
            body.append(f"🧱 What blocks full research: {_esc(why_blocked)}")

        body.extend(
            [
                "",
                "➡️ New research: ❌ until WC explained",
                "➡️ Existing holding: 👀 Monitor",
                "➡️ Sell signal: ❌ No",
                "",
                f"🪖 {_esc(ISSUER_LABELS.get(self.issuer_class or '', self.issuer_class or 'Defence / EPC'))}",
                "🧡 Cash flow — elevated watch",
                "",
                "<b>Working-capital checklist</b>",
            ]
        )
        for item in WC_RECONCILIATION_CHECKLIST:
            body.append(f"• {_esc(item)}")
        body.append(
            "Classify: <code>TEMPORARY_BILLING_CYCLE</code> · "
            "<code>WORKING_CAPITAL_STRESS</code> · "
            "<code>DATA_OR_SCOPE_ERROR</code> · <code>INCONCLUSIVE</code>."
        )
        body.append("ℹ️ <i>Pre-scan only.</i>")

        small = telegram_small_block(body)
        return "\n".join(header + (["", small] if small else []))


def _financials_source_label(source: str | None) -> str:
    if not source:
        return "Screener.in"
    if source.startswith("yfinance:"):
        return "Yahoo Finance (Screener data stale or missing)"
    if source.startswith("screener:"):
        return "Screener.in"
    return source


def _esc(value: object) -> str:
    import html

    return html.escape(str(value), quote=False)


def format_analyze_gate_block(result: EligibilityResult) -> str:
    """Telegram HTML when /analyze is blocked by the eligibility gate."""
    return (
        f"{result.telegram_html()}\n\n"
        "<b>Deep /analyze blocked</b> — this name failed the 3-year research gate.\n"
        "Run <code>/prescan SYMBOL</code> for the full card, fix data gaps, then retry.\n"
        "Override (not recommended): <code>/analyze force SYMBOL</code>"
    )


def _verdict_from_band(band: str, hard_status: str) -> tuple[EligibilityVerdict, bool]:
    """Legacy band→verdict helper (tests / AI fallback only). Prefer routing."""
    if hard_status == "DATA_UNAVAILABLE":
        return "DATA_UNAVAILABLE_RETRY", False
    if hard_status in ("HARD_EXCLUDE", "DATA_INSUFFICIENT"):
        if hard_status == "DATA_INSUFFICIENT":
            return "DATA_UNAVAILABLE_RETRY", False
        return "NOT_SUITABLE_FOR_3Y_RESEARCH", False
    if band in ("STRONG_CANDIDATE", "CANDIDATE"):
        return "AUTO_DEEP_ANALYSIS", True
    if band == "WATCHLIST":
        return "SECTOR_SPECIFIC_REVIEW", True
    return "HOLDING_MONITOR_ONLY", False


_STRUCTURAL_VERDICTS = frozenset(
    {
        "DATA_UNAVAILABLE_RETRY",
        "SECTOR_SPECIFIC_REVIEW",
        "HOLDING_MONITOR_ONLY",
        "NOT_SUITABLE_FOR_3Y_RESEARCH",
    }
)


_PROMOTER_HOLDING_AS_CRITICAL_RE = re.compile(
    r"promoter[_ ]?(?:pct|holding(?:_pct)?)[^.\n;,]{0,40}?\(Critical\)",
    re.IGNORECASE,
)
_LEGACY_PROMOTER_PCT_CRITICAL_RE = re.compile(
    r"promoter_pct\s+[\d.]+%\s*\(Critical\)",
    re.IGNORECASE,
)


def _sanitize_promoter_field_confusion(text: str) -> str:
    """Strip AI mistakes that treat promoter *holding* as a Critical pledge signal."""
    if not text:
        return text
    # Keep the numeric holding figure — only drop the false Critical label.
    cleaned = re.sub(
        r"promoter_pct\s+([\d.]+)%\s*\(Critical\)",
        r"promoter_holding_pct \1% (ownership concentration; informational)",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"promoter_holding_pct\s+([\d.]+)%\s*\(Critical\)",
        r"promoter_holding_pct \1% (ownership concentration; informational)",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = _LEGACY_PROMOTER_PCT_CRITICAL_RE.sub(
        "promoter_holding_pct (ownership; informational — not Critical)",
        cleaned,
    )
    cleaned = _PROMOTER_HOLDING_AS_CRITICAL_RE.sub(
        "promoter_holding_pct (ownership; informational — not Critical)",
        cleaned,
    )
    return cleaned


def _deterministic_weak_fundamentals_copy(
    *,
    quant_score: float,
    band: str,
    ocf_to_pat: float | None,
    red_flag_codes: list[str],
) -> tuple[str, str]:
    bits: list[str] = [f"quant_score {quant_score:.2f} ({band})"]
    if ocf_to_pat is not None and ocf_to_pat < 0.5:
        bits.append(f"OCF/PAT {ocf_to_pat:.2f} (Critical)")
    if any(c == "OCF_PAT_GAP" for c in red_flag_codes):
        bits.append("material cash-conversion red flag")
    elif red_flag_codes:
        bits.append("non-valuation component weaknesses")
    reason = "; ".join(bits)
    if ocf_to_pat is not None and ocf_to_pat < 0.5:
        risk = (
            "Persistent weak cash conversion — reported profits are not translating "
            "adequately into operating cash flow"
        )
    else:
        risk = "Insufficient quality / score for expensive deep analysis"
    return reason, risk


def _ask_eligibility_ai(
    *,
    payload: dict[str, Any],
    config: ScreenerRunConfig,
    tracker: ScreenerCostTracker,
) -> dict[str, Any] | None:
    resolved = resolve_ai_ranker(config)
    if resolved is None:
        return None
    provider, model = resolved
    system = load_eligibility_prompt()
    user_msg = (
        "Decide deep-analysis eligibility for this ONE stock.\n"
        + json.dumps(payload, indent=2)
    )

    if provider in ("openai", "deepseek"):
        api_key = (
            settings.openai_api_key if provider == "openai" else settings.deepseek_api_key
        )
        base = OPENAI_API_BASE if provider == "openai" else DEEPSEEK_API_BASE
        response = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 1024,
                "temperature": 0,
            },
            timeout=90.0,
        )
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        cached = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss = max(0, in_tok - cached)
        cost = log_call(
            model=model,
            input_tokens=miss,
            output_tokens=out_tok,
            cached_tokens=cached,
            provider=provider,
            called_at=datetime.now(UTC),
        )
        tracker.record_ai_call(input_tokens=in_tok, output_tokens=out_tok, cost_inr=cost)
        parsed = json.loads(raw)
        parsed["_provider"] = provider
        parsed["_model"] = model
        parsed["_cost_inr"] = cost
        return parsed

    # anthropic
    client = Anthropic(api_key=settings.anthropic_api_key)
    from stockbot.llm.client import call_anthropic_and_log

    response, cost = call_anthropic_and_log(
        client,
        stage="portfolio_screener_eligibility",
        ticker=str(payload.get("ticker", "ONE")),
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": system}],
        messages=[{"role": "user", "content": user_msg}],
        stream=False,
    )
    text = "\n".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    )
    usage = response.usage
    tracker.record_ai_call(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cost_inr=cost,
    )
    parsed = json.loads(text)
    parsed["_provider"] = "anthropic"
    parsed["_model"] = model
    parsed["_cost_inr"] = cost
    return parsed


def check_deep_analysis_eligibility(
    query: str,
    *,
    config: ScreenerRunConfig | None = None,
    skip_ai: bool | None = None,
) -> EligibilityResult:
    """Resolve one ticker and decide if deep master-prompt analysis is worth it."""
    config = config or ScreenerRunConfig()
    if skip_ai is not None:
        config = replace(config, skip_ai=skip_ai, dry_run=config.dry_run or skip_ai)

    table = load_symbol_table()
    resolved = resolve_ticker(query, table)
    if resolved is None:
        return EligibilityResult(
            query=query,
            ticker=None,
            company_name=None,
            verdict="NOT_FOUND",
            suitable_for_deep_analysis=False,
            key_reason="Ticker not found in NSE equity list",
        )
    if isinstance(resolved, AmbiguousMatch):
        return EligibilityResult(
            query=query,
            ticker=None,
            company_name=None,
            verdict="AMBIGUOUS",
            suitable_for_deep_analysis=False,
            ambiguous_candidates=[c.symbol for c in resolved.candidates],
            key_reason="Multiple matches — pick an exact symbol",
        )

    ticker: TickerInfo = resolved
    tracker = ScreenerCostTracker(universe_size=1)
    metrics_list = fetch_universe_metrics([ticker])
    tracker.record_stock_processed()
    m = metrics_list[0]
    quant = compute_quant_score(m, config)

    routing = decide_eligibility_route(m, quant)
    band = candidate_band(quant.final_quant_score, config.constraints)
    verdict: EligibilityVerdict = routing.eligibility  # type: ignore[assignment]
    suitable = routing.suitable_for_deep_analysis
    key_reason = routing.key_reason
    key_risk = routing.key_risk
    flag_codes = [f.code for f in quant.red_flags]
    if (
        verdict in ("HOLDING_MONITOR_ONLY", "NOT_SUITABLE_FOR_3Y_RESEARCH")
        and quant.hard_filter.status == "PASS"
        and routing.cash_conversion.status == "CRITICAL"
    ):
        key_reason, key_risk = _deterministic_weak_fundamentals_copy(
            quant_score=quant.final_quant_score,
            band=band,
            ocf_to_pat=m.ocf_to_pat,
            red_flag_codes=flag_codes,
        )

    ai_score: float | None = None
    final = quant.final_quant_score
    provider = model = None
    cost = 0.0
    concerns = list(quant.data_validation.contradictions)
    concerns.extend(governance_notes(m))
    concerns.append(f"issuer_class: {routing.issuer_class}")
    concerns.append(f"cash_conversion: {routing.cash_conversion.status}")

    # AI only for plain non-financial AUTO_DEEP — never override sector/structural routes.
    allow_ai = (
        not config.skip_ai
        and not config.dry_run
        and quant.hard_filter.status == "PASS"
        and verdict == "AUTO_DEEP_ANALYSIS"
        and routing.issuer_class in {"NON_FINANCIAL", "OTHER"}
        and not routing.quality_override
        and routing.route == "AUTO_DEEP"
    )

    if allow_ai:
        payload = {
            "ticker": quant.ticker,
            "data_timestamp": m.data_timestamp.isoformat() if m.data_timestamp else None,
            "sector": quant.sector,
            "industry": quant.industry,
            "issuer_class": routing.issuer_class,
            "eligibility_route": routing.route,
            "cash_conversion_status": routing.cash_conversion.status,
            "cash_conversion_reason": routing.cash_conversion.reason,
            "ocf_pat_current": routing.cash_conversion.ocf_pat_current,
            "ocf_pat_3y": routing.cash_conversion.ocf_pat_3y,
            "quality_override_eligible": routing.quality_override,
            "routing_eligibility": routing.eligibility,
            "market_cap_cr": m.market_cap_cr,
            "years_available": m.years_available,
            "hard_filter_status": quant.hard_filter.status,
            "hard_filter_reasons": quant.hard_filter.reasons,
            "quant_score": quant.final_quant_score,
            "base_score": quant.base_score,
            "red_flag_penalty": quant.red_flag_penalty,
            "candidate_band": band,
            "components": {
                "business_quality": quant.components.business_quality,
                "financial_strength": quant.components.financial_strength,
                "growth": quant.components.growth,
                "growth_trend": quant.components.growth_trend,
                "cash_flow_quality": quant.components.cash_flow_quality,
                "capital_efficiency": quant.components.capital_efficiency,
                "valuation": quant.components.valuation,
                "valuation_risk": quant.components.valuation_risk,
                "balance_sheet": quant.components.balance_sheet,
                "earnings_quality": quant.components.earnings_quality,
                "risk": quant.components.risk,
            },
            "metrics": {
                "roe": m.roe,
                "roe_source": m.metric_sources.get("roe"),
                "roce": m.roce,
                "roce_source": m.metric_sources.get("roce"),
                "debt_equity": m.debt_equity,
                "debt_equity_source": m.metric_sources.get("debt_equity"),
                "net_debt_ebitda": m.net_debt_ebitda,
                "net_debt_ebitda_source": m.metric_sources.get("net_debt_ebitda"),
                "interest_coverage": m.interest_coverage,
                "interest_coverage_source": m.metric_sources.get("interest_coverage"),
                "ocf_to_pat": m.ocf_to_pat,
                "ocf_to_pat_source": m.metric_sources.get("ocf_to_pat"),
                "revenue_cagr_3y": m.revenue_cagr_3y,
                "eps_cagr_3y": m.eps_cagr_3y,
                "pe": m.pe,
                "pb": m.pb,
                "ev_ebitda": m.ev_ebitda,
                "promoter_holding_pct": m.promoter_holding_pct,
                "pledged_promoter_holding_pct": m.pledged_promoter_holding_pct,
            },
            "governance_notes": governance_notes(m),
            "promoter_field_rules": {
                "promoter_holding_pct": "Informational ownership % — NEVER Critical for being high",
                "pledged_promoter_holding_pct": "Pledge risk only — Prefer ≤10, Borderline 10-25, Critical >25",
            },
            "data_confidence": quant.data_validation.data_confidence,
            "data_completeness": quant.data_validation.data_completeness_score,
            "contradictions": quant.data_validation.contradictions,
            "red_flags": [f"{f.severity}:{f.code}:{f.message}" for f in quant.red_flags],
            "rules": {
                "quant_lt_60_not_auto_reject": True,
                "single_year_ocf_pat_not_critical_alone": True,
                "banks_skip_de_ocf_interest_cover": True,
            },
        }
        try:
            ai = _ask_eligibility_ai(payload=payload, config=config, tracker=tracker)
        except Exception:
            logger.exception("eligibility AI failed for %s — using routing only", ticker.symbol)
            ai = None
        if ai:
            provider = str(ai.get("_provider"))
            model = str(ai.get("_model"))
            cost = float(ai.get("_cost_inr") or 0.0)
            try:
                ai_score = float(ai.get("ai_score", quant.final_quant_score))
            except (TypeError, ValueError):
                elig = str(ai.get("eligibility", "")).upper()
                mapped = _normalize_verdict(elig)
                ai_score = {
                    "AUTO_DEEP_ANALYSIS": 80.0,
                    "SECTOR_SPECIFIC_REVIEW": 72.0,
                    "HOLDING_MONITOR_ONLY": 45.0,
                    "NOT_SUITABLE_FOR_3Y_RESEARCH": 35.0,
                    "DATA_UNAVAILABLE_RETRY": 0.0,
                }.get(mapped or "", quant.final_quant_score)
            final = combine_scores(quant.final_quant_score, ai_score, config)
            band = candidate_band(final, config.constraints)
            # Re-route after AI blend so quant<60 + quality still wins
            quant_blended = replace(quant, final_quant_score=final)
            routing = decide_eligibility_route(m, quant_blended)
            verdict = routing.eligibility  # type: ignore[assignment]
            suitable = routing.suitable_for_deep_analysis
            key_reason = routing.key_reason
            key_risk = routing.key_risk

            # AI may only annotate reasons on AUTO_DEEP; never override structural routes.
            if verdict == "AUTO_DEEP_ANALYSIS":
                ai_reason = _sanitize_promoter_field_confusion(
                    str(ai.get("key_reason") or "")
                )
                ai_risk = _sanitize_promoter_field_confusion(str(ai.get("key_risk") or ""))
                if ai_reason:
                    key_reason = ai_reason
                if ai_risk:
                    key_risk = ai_risk

            if (
                verdict in ("HOLDING_MONITOR_ONLY", "NOT_SUITABLE_FOR_3Y_RESEARCH")
                and quant.hard_filter.status == "PASS"
                and routing.cash_conversion.status == "CRITICAL"
            ):
                key_reason, key_risk = _deterministic_weak_fundamentals_copy(
                    quant_score=final,
                    band=band,
                    ocf_to_pat=m.ocf_to_pat,
                    red_flag_codes=flag_codes,
                )
            extra = ai.get("data_concerns") or []
            if isinstance(extra, list):
                for c in extra:
                    cleaned = _sanitize_promoter_field_confusion(str(c))
                    low = cleaned.lower()
                    if "promoter_pct" in low and "critical" in low and "pledge" not in low:
                        continue
                    if cleaned and cleaned not in concerns:
                        concerns.append(cleaned)

    recheck_note: str | None = None
    # Safety net: fetch/data gaps are never quality rejects.
    if fundamentals_fetch_failed(m) or quant.hard_filter.status == "DATA_UNAVAILABLE":
        verdict = "DATA_UNAVAILABLE_RETRY"
        suitable = False
        routing = decide_eligibility_route(m, quant)
        key_reason = routing.key_reason
        key_risk = routing.key_risk
        recheck_note = (
            "Re-check after fetch retry — missing/empty fundamentals are not a "
            "quality conclusion"
        )
    elif (
        quant.hard_filter.status == "DATA_INSUFFICIENT"
        and verdict != "DATA_UNAVAILABLE_RETRY"
    ):
        verdict = "DATA_UNAVAILABLE_RETRY"
        suitable = False
        key_reason = "; ".join(quant.hard_filter.reasons) or "critical metrics missing"
        key_risk = "Incomplete data — retry before concluding 3y research eligibility"
        missing_keys = [
            k
            for k in ("roe", "debt_equity", "ocf_to_pat", "eps", "revenue", "net_income")
            if k in quant.data_validation.missing_metrics
        ]
        if len(missing_keys) <= 2:
            recheck_note = (
                "Re-check in ~30 days — Screener/NSE may fill missing ratios "
                f"({', '.join(missing_keys) or 'critical fields'})"
            )

    # Surface computed-metric provenance as concerns (transparency)
    for field_name, source in m.metric_sources.items():
        if source in ("computed", "yfinance"):
            note = f"{field_name} source: {source}"
            if note not in concerns:
                concerns.append(note)

    computed_warnings = format_computed_metric_warnings(
        m.metric_sources,
        {
            "roe": m.roe,
            "roce": m.roce,
            "debt_equity": m.debt_equity,
            "ocf_to_pat": m.ocf_to_pat,
            "interest_coverage": m.interest_coverage,
            "net_debt_ebitda": m.net_debt_ebitda,
            "current_ratio": m.current_ratio,
            "operating_margin": m.operating_margin,
            "ebitda_margin": m.ebitda_margin,
            "asset_turnover": m.asset_turnover,
        },
    )

    result = EligibilityResult(
        query=query,
        ticker=ticker.symbol,
        company_name=ticker.company_name,
        verdict=verdict,
        suitable_for_deep_analysis=suitable,
        quant_score=round(quant.final_quant_score, 2),
        ai_score=round(ai_score, 2) if ai_score is not None else None,
        final_score=round(final, 2),
        candidate_band=band,
        price_at_scan=(
            round(m.current_price_abs, 2) if m.current_price_abs is not None else None
        ),
        hard_filter_status=quant.hard_filter.status,
        hard_filter_reasons=list(quant.hard_filter.reasons),
        sector=quant.sector,
        issuer_class=routing.issuer_class,
        eligibility_route=routing.route,
        cash_conversion_status=routing.cash_conversion.status,
        cash_conversion_interpretation=routing.cash_conversion.interpretation or None,
        ocf_pat_current=(
            round(routing.cash_conversion.ocf_pat_current, 2)
            if routing.cash_conversion.ocf_pat_current is not None
            else None
        ),
        ocf_pat_3y_cumulative=(
            round(routing.cash_conversion.ocf_pat_3y, 2)
            if routing.cash_conversion.ocf_pat_3y is not None
            else None
        ),
        ocf_current_abs=(
            round(routing.cash_conversion.ocf_current, 1)
            if routing.cash_conversion.ocf_current is not None
            else None
        ),
        cfo_3y_sum_abs=(
            round(routing.cash_conversion.cfo_3y_sum, 1)
            if routing.cash_conversion.cfo_3y_sum is not None
            else None
        ),
        pat_3y_sum_abs=(
            round(routing.cash_conversion.pat_3y_sum, 1)
            if routing.cash_conversion.pat_3y_sum is not None
            else None
        ),
        debt_equity=round(m.debt_equity, 2) if m.debt_equity is not None else None,
        interest_coverage=(
            round(m.interest_coverage, 2) if m.interest_coverage is not None else None
        ),
        net_debt_ebitda=(
            round(m.net_debt_ebitda, 2) if m.net_debt_ebitda is not None else None
        ),
        roe=round(m.roe, 2) if m.roe is not None else None,
        next_research_action=routing.next_action,
        quality_override=routing.quality_override,
        quality_score=round(quant.components.business_quality, 1),
        growth_score=round(quant.components.growth, 1),
        valuation_score=round(quant.components.valuation, 1),
        financial_strength_score=round(quant.components.financial_strength, 1),
        risk_score=round(quant.components.risk, 1),
        data_confidence=quant.data_validation.data_confidence,
        data_completeness=quant.data_validation.data_completeness_score,
        financials_basis=m.financials_basis,
        financials_source=m.financials_source,
        sector_source=m.sector_source,
        derived_metric_count=count_derived_key_ratios(m),
        key_reason=key_reason,
        key_risk=key_risk,
        data_concerns=concerns,
        computed_metric_warnings=computed_warnings,
        recheck_note=recheck_note,
        ai_provider=provider,
        ai_model=model,
        cost_inr=round(cost, 2),
    )

    missing_key = [
        k
        for k in ("roe", "debt_equity", "ocf_to_pat")
        if k in quant.data_validation.missing_metrics
    ]
    log_prescan_outcome(
        {
            "ticker": ticker.symbol,
            "verdict": verdict,
            "suitable_for_deep_analysis": suitable,
            "issuer_class": routing.issuer_class,
            "eligibility_route": routing.route,
            "cash_conversion_status": routing.cash_conversion.status,
            "quality_override": routing.quality_override,
            "next_research_action": routing.next_action,
            "ocf_pat_3y_cumulative": routing.cash_conversion.ocf_pat_3y,
            "hard_filter_status": quant.hard_filter.status,
            "hard_filter_reasons": list(quant.hard_filter.reasons),
            "quant_score": round(quant.final_quant_score, 2),
            "candidate_band": band,
            "price_at_scan": (
                round(m.current_price_abs, 2) if m.current_price_abs is not None else None
            ),
            "reject_class": classify_reject(
                hard_status=quant.hard_filter.status,
                verdict=verdict,
                quant_score=quant.final_quant_score,
            ),
            "computed_metrics": {
                k: v for k, v in m.metric_sources.items() if v in ("computed", "yfinance")
            },
            "metric_sources": dict(m.metric_sources),
            "financials_basis": m.financials_basis,
            "sector_source": m.sector_source,
            "derived_metric_count": count_derived_key_ratios(m),
            "missing_key_trio": missing_key,
            "data_completeness": quant.data_validation.data_completeness_score,
            "data_confidence": quant.data_validation.data_confidence,
            "quality_score": round(quant.components.business_quality, 1),
            "growth_score": round(quant.components.growth, 1),
            "strength_score": round(quant.components.financial_strength, 1),
        }
    )
    return result
