"""Module 9 — Stage 2: verdict against the external master prompt.

The master prompt (prompts/master-stock-analysis-prompt-v3.md) is loaded
as the Stage 2 system prompt. The Quality-First constitution
(prompts/quality-first-portfolio-constitution-v1.md) is prepended as
permanent top-level policy. Neither file is paraphrased into code — only
composed (constitution first, then master).

It cannot use client.messages.parse() the way Stage 1 does: the master
prompt's own format is a full 16-section prose report, and only the JSON
block layered on top of it (via the hard injections below) needs to be
structured. Forcing output_config.format would collapse the whole
response into pure JSON and lose the prose the report is actually for.
So this calls client.messages.stream() (not .create() — the SDK refuses
non-streaming calls once max_tokens is high enough that generation could
exceed 10 minutes, hit live once MAX_TOKENS reached 32000) and parses the
trailing fenced ```json block out of the assembled response text
afterward.

No `temperature` parameter — rejected outright on these models (see
PROJECT.md). Same reason `thinking={"type": "enabled", "budget_tokens": N}`
is not used here even though it was requested during the v3 migration:
Sonnet 5 rejects `budget_tokens` with a 400 (removed on that model
generation, same family as the temperature rejection). `{"type": "adaptive"}`
is the correct, accepted shape and is what's used below.

Hard injections are layered into the user message (inside
<pipeline_constraints>) rather than edited into the master prompt file
itself. They reinforce rules already stated natively by v3:
  1. Technical figures are code-computed [FACT], cite [PRICE_AND_TECHNICALS].
  2. Confidence is 1–10 but capped at 7/10 for this pipeline — always X/10.
  3. No web search — retrieval already ran; closed-world + explicit "cannot
     be determined" behaviour.
  4. Source-conflict preference order.
Plus the trailing JSON schema (valid JSON only; null for incomplete fields).
"""

from __future__ import annotations

import json
import re
from datetime import date

from anthropic import Anthropic
from pydantic import BaseModel, Field

from stockbot.analysis.analysis_context import (
    format_peer_snapshot_json,
    format_portfolio_execution_json,
    format_sector_scorecard_json,
)
from stockbot.brief import (
    format_financials_section,
    format_price_section,
    format_shareholding_section,
)
from stockbot.brief_enrichment import (
    format_metadata_json,
    format_news_summary_json,
    format_prescan_summary_json,
)
from stockbot.config import MASTER_PROMPT_PATH, PROMPTS_DIR, settings
from stockbot.fetch.annual_report import (
    BUSINESS_HEADING_PRIORITY,
    business_narrative_gap,
    format_ar_business_summary_json,
)

CONSTITUTION_PATH = PROMPTS_DIR / "quality-first-portfolio-constitution-v1.md"
STAGE2_LITE_PROMPT_PATH = PROMPTS_DIR / "stage2-lite-v1.md"
from stockbot.analysis_routing import Stage2Mode
from stockbot.llm.client import call_anthropic_and_log
from stockbot.llm.extract import ExtractionResult
from stockbot.models import Brief
from stockbot.order_book_signals import (
    collect_order_book_signals,
    format_order_book_signals_for_stage2,
    order_book_wc_billing_hint,
)

MODEL = "claude-sonnet-5"
LITE_MODEL = "claude-haiku-4-5-20251001"
# v3 migration: switched from claude-opus-5. Opus goes back in only after
# a clean end-to-end run on Sonnet, per the migration's own instruction.
#
# max_tokens history, each raise forced by a real truncation: 8000 (the
# migration's starting value) truncated TCS with ZERO characters generated
# — the whole budget consumed by adaptive thinking before any report text.
# 16000 then truncated TCS again (8,594 chars), and truncated JYOTHYLAB on
# all 3 attempts and INFY on all 3 attempts in the same live smoke-test run
# — only IRCTC passed, and its annual-report fetch had timed out, giving it
# less context and a shorter generation. That pattern (full annual-report
# context -> consistently exceeds 16000) is a strong signal, not noise.
# 32000 still truncated long FULL utilities runs (thinking + 16 sections).
# Base 48000 with one escalate to 64000: Sonnet 5 allows up to 128K output,
# but ~64K is the practical ceiling under the ₹80 per-analysis cap once
# Stage 1 is paid. Unused max_tokens are not billed — only actual output.
MAX_TOKENS = 48_000
# Model hard caps (Anthropic docs): Haiku 4.5 = 64K, Sonnet 5 = 128K.
# LITE: 4096 truncated HBLENGINE/GESHIP 3×; 8192 interim; 16384 still
# truncated fat briefs on first attempt (health audit 2026-09-02: GESHIP
# 17 calls / HBLENGINE orphan session). Start at 32768 so the common case
# finishes without a billed truncation retry; escalate once to the Haiku cap.
LITE_MAX_TOKENS = 32_768
LITE_MAX_TOKENS_CAP = 65_536
MAX_TOKENS_CAP = 64_000


def stage2_max_tokens(mode: Stage2Mode, truncation_attempt: int = 0) -> int:
    """Output budget for one Stage 2 call. Escalates on truncation retries.

    Retrying the same max_tokens after stop_reason=max_tokens is pure waste —
    same prompt, same ceiling, same failure. Each attempt must raise the
    budget (or already sit at the mode cap).
    """
    if truncation_attempt < 0:
        raise ValueError(f"truncation_attempt must be >= 0, got {truncation_attempt}")
    if mode == "LITE":
        ladder = (LITE_MAX_TOKENS, LITE_MAX_TOKENS_CAP)
    else:
        ladder = (MAX_TOKENS, MAX_TOKENS_CAP)
    return ladder[min(truncation_attempt, len(ladder) - 1)]

PIPELINE_CONFIDENCE_CAP = 7

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def effective_confidence_cap(brief: Brief) -> int:
    """Min of the pipeline-wide 7 cap and the brief's data-driven ceiling."""
    return min(PIPELINE_CONFIDENCE_CAP, brief.confidence_ceiling)


from stockbot.trade_policy import trade_friendly_mode


def _constitution_rule_9_text() -> str:
    if trade_friendly_mode():
        return (
            "9. Quality-First constitution (trade-friendly mode ON): complete "
            "five_year_business_test before any Ideal Buy / Add More zone. If "
            "answer is NO, withhold buy/add ranges. If UNCERTAIN, you MAY issue "
            "buy_zone_abs when evidence_for lists at least two specific, "
            "filing-backed trends (or one trend with confidence HIGH) and set "
            "buy_range_allowed=true — label the zone as thesis-conditional in "
            "§13. For defence/EPC/project names with extremely weak reported "
            "cash conversion (e.g. 3y ΣCFO/ΣPAT < 0.25, or sharply negative "
            "OCF vs profit), set wc_gap_classification; WORKING_CAPITAL_STRESS "
            "still blocks ranges. INCONCLUSIVE/DATA_OR_SCOPE_ERROR may carry a "
            "range when reconciled in prose. A lower price alone never creates "
            "an add. When a buy range is allowed, present the position_building_plan "
            "as a 4-tranche table per §13. Anti-chase pauses new capital after "
            "abnormal short-term surges unless price is already inside the issued "
            "buy zone. Profit review = rebalancing review, not an exact top."
        )
    return (
        "9. Quality-First constitution (system prompt): complete "
        "five_year_business_test before any Ideal Buy / Add More zone. If answer "
        "is NO or UNCERTAIN, set buy_range_allowed=false, add_range_allowed=false, "
        "buy_zone_abs=null, and do not invent buy/add levels. For defence/EPC/project "
        "names with extremely weak reported cash conversion (e.g. 3y ΣCFO/ΣPAT < 0.25, "
        "or sharply negative OCF vs profit), set wc_gap_classification and withhold "
        "buy/add ranges unless the classification is TEMPORARY_BILLING_CYCLE with a "
        "year-by-year CFO-to-PAT reconciliation in the report. A lower price alone "
        "never creates an add. When a buy range is allowed, present the "
        "position_building_plan as a 4-tranche table per §13 — tranche 1 may deploy "
        "at current price once the quality/five-year/anti-chase gates pass; tranches "
        "2-4 stay valuation-gated and conditional, never automatic solely because "
        "price fell. Anti-chase pauses new capital (including tranche 1) after "
        "abnormal short-term surges. Profit review = rebalancing review, not an exact top."
    )


def hard_injections(confidence_cap: int) -> str:
    rule_9 = _constitution_rule_9_text()
    return f"""
IMPORTANT — pipeline constraints, read before writing the report:

1. The technical figures below (SMA50, SMA200, RSI14, support/resistance) \
were computed from real OHLCV data in code. Treat them as [FACT] and cite \
[PRICE_AND_TECHNICALS]. Do not recompute, estimate, or second-guess them yourself.

2. Confidence is 1–10, but this pipeline caps at {confidence_cap}/10 maximum. \
Never exceed {confidence_cap} regardless of evidence quality. Always write \
confidence as X/10 — never X/{confidence_cap}. If your assessment would \
otherwise be higher, cap the number at {confidence_cap} but still describe your true \
assessment in the prose.

3. Retrieval (price, financials, shareholding, annual-report extraction, \
news / disconfirmation search) has already been performed by a separate \
data-fetching pipeline and is contained entirely in the <context> blocks \
below. You do NOT have a web search tool in this call — do not attempt to \
research further, and do not assume access to any live data beyond what is \
provided. Treat the provided material as the complete evidence base; any \
gap is a genuine finding to report as MISSING or [UNVERIFIED], never \
something to fill from general knowledge. If analysis cannot be completed \
from context, state "This cannot be determined from the supplied evidence."

4. On source conflicts: prefer PRICE_AND_TECHNICALS over all for price/tech; \
FINANCIALS over EXTRACTION/news for accounting numbers; SHAREHOLDING over \
news for ownership/pledge. Note conflicts in §6.

5. Use <metadata> sector/industry and <prescan_summary> issuer_class to choose \
the sector-appropriate scorecard (bank, defence/EPC, utility, loss-making growth). \
Report P/E as metadata.pe_price_eps — it is computed as this company's own price \
÷ TTM EPS from the same FINANCIALS table you cite, so it always matches the price \
and EPS you state elsewhere in the report. metadata.ttm_pe is Yahoo's separately \
sourced trailing P/E snapshot and can lag or use a different EPS base; mention it \
only as a secondary cross-check if it differs materially from pe_price_eps, never \
as the primary reported multiple. If pe_price_eps is null (EPS unavailable or \
non-positive), state P/E as MISSING rather than inventing one or silently \
substituting ttm_pe. Do not invent forward P/E or sector median multiples. If \
metadata.street_consensus is present, use it only as an external tension check \
(price vs mean target) — never as the primary thesis or fair-value anchor.

6. You may use <prescan_summary> as a starting point for routing labels \
(e.g. DEFENCE_WC_REVIEW) and multi-year cash-conversion context, but verify \
underlying numbers in FINANCIALS. Do not change issuer_class or route unless \
detailed data clearly contradicts them.

7. Use <news_summary> only to supplement fundamentals (order book headlines, \
management changes, guidance). Do not treat broker targets or headlines as facts \
without cross-checking filings/results.

8. Use <ar_business_summary> for filing-sourced order-book / segment / MD&A \
highlights when present. Prefer these over news for backlog size; mark anything \
not directly supported in FINANCIALS as [UNVERIFIED] in prose.

9. {rule_9}

10. Read <data_inventory> before §15A. If pipeline_missing is empty and \
FINANCIALS span multiple years, base five_year_business_test on those \
numbers. Do not answer UNCERTAIN solely because MD&A is thin when FINANCIALS \
show a clear multi-year trend — name the specific trend in evidence_for or \
evidence_against. Use UNCERTAIN only for genuinely mixed business evidence, \
not pipeline gaps. When buy_range_allowed is false, state the gate in §13 \
and Beginner Summary (e.g. "five-year: UNCERTAIN — margin compression") — \
never vague "needs more evidence" without naming the gate.

At the very end of your report, after the Final Beginner Summary, include a \
fenced ```json code block containing exactly these fields. Output only valid \
JSON — if a field cannot be completed, use null (or [] for arrays). The numbers must \
be the SAME as used in the prose above — this is a restatement for the \
pipeline to parse, never a re-derivation:

```json
{{
  "verdict": "BUY|BUY ON CORRECTION|WATCH|SKIP",
  "current_price_abs": <number>,
  "price_date": "<YYYY-MM-DD>",
  "buy_zone_abs": [<low>, <high>] | null,
  "valuation_inputs": {{
    "eps_bear": <number>, "eps_base": <number>, "eps_bull": <number>,
    "multiple_bear": [<low>, <high>],
    "multiple_base": [<low>, <high>],
    "multiple_bull": [<low>, <high>]
  }},
  "confidence": <integer, 1-10 scale, capped at {confidence_cap} — always written as X/10>,
  "risk": "LOW|MEDIUM|HIGH",
  "business_quality": <integer 1-10>,
  "financial_health": <integer 1-10>,
  "management_quality": <integer 1-10>,
  "earnings_quality": "HIGH|MEDIUM|LOW",
  "holding_period": "<string>",
  "reasons_buy": ["<string>", "..."],
  "reasons_avoid": ["<string>", "..."],
  "biggest_watch": "<string>",
  "missing_data_impact": "<string>",
  "gates_failed": ["<string>", "..."],
  "bear_growth_justification": null,
  "five_year_business_test": {{
    "answer": "YES|NO|UNCERTAIN",
    "confidence": "HIGH|MEDIUM|LOW",
    "evidence_for": [],
    "evidence_against": []
  }},
  "buy_range_allowed": <true|false>,
  "add_range_allowed": <true|false>,
  "thesis_status": null,
  "anti_chase_flag": <true|false>,
  "thesis_invalidation_triggers": [],
  "wc_gap_classification": null,
  "profit_review": {{
    "status": "NOT_TRIGGERED|REVIEW_FOR_REBALANCING",
    "trigger_reason": [],
    "note": "A valuation-range review is not an automatic sell instruction."
  }},
  "position_building_plan": null,
  "expected_return": {{
    "horizon_years": 3,
    "assumptions": ["<string>", "..."],
    "confidence": "HIGH|MEDIUM|LOW",
    "note": "<probabilistic disclaimer — no guaranteed yearly ladder>"
  }}
}}
```

Expected return rules (mandatory when valuation_inputs are present):
- Do **not** state fixed yearly return ladders (forbidden: "year 1 = 12%, year 2 = 14%").
- Supply only horizon_years (2–5), assumptions (EPS/multiple/order-book/cash-flow drivers), \
confidence, and note in JSON — Python computes bear/base/bull **CAGR ranges** from \
fair-value scenarios vs current price.
- If buy_range_allowed is false or thesis/WC gates are unresolved, assumptions must \
reflect uncertainty; ranges are **educational only**, not actionable targets.
- Never invent broker/consensus figures not supported in supplied context — mark \
external forecasts [UNVERIFIED] in prose and assumptions.

Note on valuation_inputs: supply an EPS estimate and a P/E multiple RANGE
for each of bear/base/bull — never a price you multiplied yourself.
Python computes the resulting fair-value ranges from these.
"""


# Back-compat alias for imports/tests that still reference the constant name.
HARD_INJECTIONS = hard_injections(PIPELINE_CONFIDENCE_CAP)

class VerdictParseError(Exception):
    pass


class TruncatedResponseError(Exception):
    """Distinct from VerdictParseError on purpose: a truncated response is
    an infrastructure failure (max_tokens too low for this call), not a
    validation failure. pipeline.py must not let it consume one of the
    scarce Stage 2 validation retries — see the v3 migration note there."""

    def __init__(self, cost_inr: float, char_count: int, max_tokens: int):
        self.cost_inr = cost_inr
        super().__init__(
            f"Stage 2 response was truncated at max_tokens={max_tokens} before completing "
            f"(stop_reason='max_tokens'). {char_count} chars generated, cost ₹{cost_inr:.2f} "
            f"still logged. This is an infrastructure failure, not a validation failure."
        )


class ValuationInputs(BaseModel):
    eps_bear: float
    eps_base: float
    eps_bull: float
    multiple_bear: tuple[float, float]
    multiple_base: tuple[float, float]
    multiple_bull: tuple[float, float]


class FiveYearBusinessTest(BaseModel):
    answer: str
    confidence: str | None = None
    evidence_for: list[str] = []
    evidence_against: list[str] = []


class ProfitReview(BaseModel):
    status: str = "NOT_TRIGGERED"
    trigger_reason: list[str] = []
    note: str | None = None


class ExpectedReturnInputs(BaseModel):
    """Model narrative only — CAGR ranges are computed in Python (expected_return.py)."""

    horizon_years: int = 3
    assumptions: list[str] = Field(default_factory=list)
    confidence: str = "MEDIUM"
    note: str | None = None


class VerdictJSON(BaseModel):
    verdict: str
    current_price_abs: float
    price_date: date
    # null when constitution gates block a buy/add range (five-year test
    # NO/UNCERTAIN, quality fail, or thesis invalidation).
    buy_zone_abs: tuple[float, float] | None = None
    valuation_inputs: ValuationInputs
    confidence: int
    risk: str
    business_quality: int
    financial_health: int
    management_quality: int
    earnings_quality: str
    holding_period: str
    reasons_buy: list[str]
    reasons_avoid: list[str]
    biggest_watch: str
    missing_data_impact: str
    gates_failed: list[str]
    # Bear-case calibration (17A/17B): required only when eps_bear exceeds
    # trailing EPS — otherwise omitted/null. Optional so existing reports
    # (bear EPS at or below TTM, the normal case) don't need to supply it.
    bear_growth_justification: str | None = None
    # Quality-First constitution fields — optional so older fixtures parse.
    five_year_business_test: FiveYearBusinessTest | None = None
    buy_range_allowed: bool | None = None
    add_range_allowed: bool | None = None
    thesis_status: str | None = None
    anti_chase_flag: bool | None = None
    external_valuation_tension: str | None = None
    thesis_invalidation_triggers: list[str] | None = None
    # Defence/EPC cash-gap outcome. Only TEMPORARY_BILLING_CYCLE unlocks
    # buy/add ranges when reported conversion is extremely weak.
    wc_gap_classification: str | None = None
    profit_review: ProfitReview | None = None
    position_building_plan: dict | list | None = None
    expected_return: ExpectedReturnInputs | None = None


class ValuationComputed(BaseModel):
    """Fair-value ranges computed in Python from the model's valuation_inputs
    — eps × multiple, never trusted from the model's own arithmetic. This is
    prevention, not detection: there is no model-stated price left to check
    for drift against, because the model is never asked to state one."""

    fair_value_bear_abs: tuple[float, float]
    fair_value_base_abs: tuple[float, float]
    fair_value_bull_abs: tuple[float, float]


def _price_range(eps: float, multiple: tuple[float, float]) -> tuple[float, float]:
    # A negative EPS (loss-making company — v3's own sector-adaptation rule
    # covers this case) flips the ordering of eps * multiple: sort rather
    # than assume multiple[0] * eps <= multiple[1] * eps.
    low, high = eps * multiple[0], eps * multiple[1]
    return (low, high) if low <= high else (high, low)


def compute_valuation(inputs: ValuationInputs) -> ValuationComputed:
    return ValuationComputed(
        fair_value_bear_abs=_price_range(inputs.eps_bear, inputs.multiple_bear),
        fair_value_base_abs=_price_range(inputs.eps_base, inputs.multiple_base),
        fair_value_bull_abs=_price_range(inputs.eps_bull, inputs.multiple_bull),
    )


def load_master_prompt() -> str:
    """Load constitution (top-level policy) then the Stage 2 master prompt.

    The master markdown file is not paraphrased in code; composition is
    prepend-only so the Quality-First constitution is always in force.
    """
    if not MASTER_PROMPT_PATH.exists():
        raise VerdictParseError(
            f"Master prompt not found at {MASTER_PROMPT_PATH} — this is a required "
            f"external asset, not something the pipeline can proceed without."
        )
    master = MASTER_PROMPT_PATH.read_text(encoding="utf-8")
    if CONSTITUTION_PATH.exists():
        constitution = CONSTITUTION_PATH.read_text(encoding="utf-8").strip()
        return (
            constitution
            + "\n\n---\n\n"
            + "# MASTER ANALYSIS PROTOCOL (follows constitution)\n\n"
            + master
        )
    return master


def load_lite_prompt() -> str:
    """Compact Stage 2 path — constitution + lite section template."""
    if not STAGE2_LITE_PROMPT_PATH.exists():
        raise VerdictParseError(
            f"Lite Stage 2 prompt not found at {STAGE2_LITE_PROMPT_PATH}"
        )
    lite = STAGE2_LITE_PROMPT_PATH.read_text(encoding="utf-8")
    if CONSTITUTION_PATH.exists():
        constitution = CONSTITUTION_PATH.read_text(encoding="utf-8").strip()
        return (
            constitution
            + "\n\n---\n\n"
            + "# LITE ANALYSIS PROTOCOL (follows constitution)\n\n"
            + lite
        )
    return lite


def load_stage2_system_prompt(mode: Stage2Mode) -> str:
    if mode == "LITE":
        return load_lite_prompt()
    return load_master_prompt()


def _format_extraction_result(extraction: ExtractionResult) -> str:
    lines = ["### Stage 1 Extraction (from the annual report and news)"]
    lines.append(f"- Auditor opinion type: {extraction.auditor_opinion_type or 'MISSING: not determined'}")
    if extraction.auditor_concerns:
        lines.append("- Auditor concerns:")
        lines.extend(f"  - {c}" for c in extraction.auditor_concerns)
    if extraction.key_audit_matters:
        lines.append("- Key audit matters:")
        lines.extend(f"  - {m}" for m in extraction.key_audit_matters)
    lines.append(
        f"- Related party: {extraction.related_party_summary or 'MISSING: not determined'} "
        f"(amount: {extraction.related_party_amount_cr if extraction.related_party_amount_cr is not None else 'MISSING'} ₹cr)"
    )
    lines.append(
        f"- Contingent liabilities: "
        f"{extraction.contingent_liabilities_cr if extraction.contingent_liabilities_cr is not None else 'MISSING'} ₹cr"
    )
    if extraction.red_flags_found:
        lines.append("- Red flags found in news:")
        for flag in extraction.red_flags_found:
            lines.append(f"  - [{flag.found_by_query}] {flag.headline} ({flag.published_date.isoformat()})")
    else:
        lines.append("- Red flags found in news: none")
    if extraction.extraction_gaps:
        lines.append("- Extraction gaps:")
        lines.extend(f"  - {g}" for g in extraction.extraction_gaps)
    return "\n".join(lines)


def _pledge_warning(brief: Brief) -> str | None:
    # Found live, twice in a row on the same real ticker: the generic
    # "don't use general knowledge" injection was not specific enough —
    # Opus kept stating a pledge percentage anyway (likely recalled from
    # training data about the company) even though it was never confirmed
    # by an exchange source, failing validate.py's pledge_not_invented
    # check on both the original attempt and the one retry. This is a
    # pointed, adjacent-to-the-data warning for that exact failure mode,
    # only injected when pledge is genuinely unconfirmed.
    pledge_confirmed = (
        brief.shareholding is not None
        and brief.shareholding.pledge_pct_of_promoter_holding is not None
    )
    if pledge_confirmed:
        return None
    return (
        "PLEDGE NOTE: promoter pledge could not be confirmed from an exchange source for "
        "this company (see Shareholding above — the field is blank/unconfirmed). Do NOT "
        "state any specific pledge percentage anywhere in the report, even one you believe "
        "you recall from general knowledge about this company. Write that pledge status is "
        "unconfirmed instead. This is checked automatically and the report will be rejected "
        "if a pledge percentage appears without it being confirmed above."
    )


def format_data_inventory_json(brief: Brief) -> str:
    """Structured pipeline gap inventory — shared by Stage 2 and validators."""
    ar = brief.annual_report
    business_present = [h for h in BUSINESS_HEADING_PRIORITY if h in ar.sections]
    business_dropped = [h for h in ar.dropped_sections if h in BUSINESS_HEADING_PRIORITY]
    payload: dict[str, object] = {
        "pipeline_missing": brief.missing,
        "confidence_ceiling": brief.confidence_ceiling,
        "financials_available": brief.financials is not None,
        "annual_report_sections": list(ar.sections.keys()),
        "annual_report_business_sections_present": business_present,
        "annual_report_business_sections_dropped": business_dropped,
        "ar_business_summary_present": ar.business_summary is not None,
        "prescan_data_confidence": (
            brief.prescan_summary.data_confidence if brief.prescan_summary else None
        ),
    }
    narrative_gap = business_narrative_gap(ar.sections, ar.dropped_sections)
    if narrative_gap:
        payload["business_narrative_gap"] = narrative_gap
    return json.dumps(payload, indent=2)


def build_user_message(
    brief: Brief, extraction: ExtractionResult, extra_instruction: str | None = None
) -> str:
    # XML delimiters match prompts/master-stock-analysis-prompt-v3.md
    # "EXPECTED INPUT STRUCTURE" — citation IDs for the model.
    context_parts: list[str] = [
        "<context>",
        "<metadata>",
        format_metadata_json(brief.metadata, brief.street_consensus),
        "</metadata>",
        "",
        "<prescan_summary>",
        format_prescan_summary_json(brief.prescan_summary),
        "</prescan_summary>",
        "",
        "<data_inventory>",
        format_data_inventory_json(brief),
        "</data_inventory>",
        "",
        "<price_and_technicals>",
        format_price_section(brief.price, brief.technicals),
        "</price_and_technicals>",
        "",
        "<financials>",
        format_financials_section(brief.financials),
        "</financials>",
        "",
        "<peer_fundamentals>",
        format_peer_snapshot_json(brief.peer_snapshot),
        "</peer_fundamentals>",
        "",
        "<sector_scorecard>",
        format_sector_scorecard_json(brief.sector_scorecard),
        "</sector_scorecard>",
        "",
        "<shareholding>",
        format_shareholding_section(brief.shareholding),
        "</shareholding>",
        "",
        "<news_summary>",
        format_news_summary_json(brief.news_summary),
        "</news_summary>",
        "",
        "<ar_business_summary>",
        format_ar_business_summary_json(brief.annual_report.business_summary),
        "</ar_business_summary>",
        "",
        "<portfolio_execution>",
        format_portfolio_execution_json(brief.portfolio_execution),
        "</portfolio_execution>",
    ]
    pledge_note = _pledge_warning(brief)
    if pledge_note:
        context_parts.extend(["", "<pipeline_note>", pledge_note, "</pipeline_note>"])
    order_book_signals = collect_order_book_signals(brief)
    if order_book_signals:
        context_parts.extend(
            [
                "",
                "<external_claims>",
                "Order-book / backlog signals (news = UNVERIFIED; annual report = primary filing excerpt):",
                *[f"- {line}" for line in format_order_book_signals_for_stage2(order_book_signals)],
                "Treat news as external claims only; reconcile against investor presentation / results.",
                "</external_claims>",
            ]
        )
        wc_hint = order_book_wc_billing_hint(brief, order_book_signals)
        if wc_hint:
            context_parts.extend(["", "<pipeline_note>", wc_hint, "</pipeline_note>"])
    context_parts.extend(
        [
            "",
            "<extraction>",
            _format_extraction_result(extraction),
            "</extraction>",
            "",
            "<pipeline_constraints>",
            hard_injections(effective_confidence_cap(brief)).strip(),
            "</pipeline_constraints>",
        ]
    )
    if extra_instruction:
        context_parts.extend(
            [
                "",
                "<retry_feedback>",
                extra_instruction,
                "</retry_feedback>",
            ]
        )
    context_parts.append("</context>")

    company_label = f"{brief.ticker.company_name} ({brief.ticker.symbol}, {brief.ticker.exchange})"
    context_parts.extend(
        [
            "",
            "<instruction>",
            f"Analyze: {company_label}",
            "</instruction>",
        ]
    )
    return "\n".join(context_parts)


def extract_verdict_json(report_text: str) -> VerdictJSON:
    matches = _JSON_BLOCK_RE.findall(report_text)
    if not matches:
        raise VerdictParseError("No fenced ```json block found in the Stage 2 report")

    raw = matches[-1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerdictParseError(f"Stage 2 JSON block is not valid JSON: {exc}") from exc

    try:
        return VerdictJSON.model_validate(data)
    except Exception as exc:
        raise VerdictParseError(f"Stage 2 JSON block failed schema validation: {exc}") from exc


def run_stage2(
    brief: Brief,
    extraction: ExtractionResult,
    client: Anthropic | None = None,
    extra_instruction: str | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    mode: Stage2Mode = "FULL",
    *,
    enable_thinking: bool | None = None,
) -> tuple[str, VerdictJSON, dict]:
    # model override exists only for pre-Opus cost-conscious debugging (e.g.
    # running the real master prompt + hard injections against Sonnet 5,
    # ~5x cheaper, to check the prompt itself works before ever spending on
    # Opus) — production calls always take the default (Opus 5).
    client = client or Anthropic(api_key=settings.anthropic_api_key)
    if mode == "LITE":
        call_model = model or LITE_MODEL
        call_max_tokens = max_tokens or LITE_MAX_TOKENS
        thinking = None
    else:
        call_model = model or settings.stage2_full_model or MODEL
        call_max_tokens = max_tokens or MAX_TOKENS
        use_thinking = (
            settings.stage2_full_thinking if enable_thinking is None else enable_thinking
        )
        if use_thinking is False:
            thinking = None
        else:
            # {"type": "adaptive"} is the only accepted shape on Sonnet 5 —
            # budget_tokens is rejected outright (400), not just deprecated.
            thinking = {"type": "adaptive"}
    system_prompt = load_stage2_system_prompt(mode)
    user_message = build_user_message(brief, extraction, extra_instruction)

    # Streaming, not .create(): the Anthropic SDK refuses a non-streaming
    # call once max_tokens is high enough that generation could plausibly
    # exceed 10 minutes ("Streaming is required for operations that may
    # take longer than 10 minutes") — hit live once max_tokens reached
    # 32000. get_final_message() returns the same Message shape .create()
    # would have, so nothing below this needs to know the difference.
    response, cost_inr = call_anthropic_and_log(
        client,
        stage="stage2_lite" if mode == "LITE" else "stage2",
        ticker=brief.ticker.symbol,
        model=call_model,
        max_tokens=call_max_tokens,
        # system_prompt is byte-identical on every call regardless of
        # ticker — cache it. 1h TTL (not the 5m default) on purpose: real
        # cache_creation_tokens observed live (~8-9k tokens per write), but
        # the 5-min window barely helps this bot's actual usage pattern —
        # single, infrequent analyses spaced well apart, per the caching
        # discussion earlier tonight. 1h covers the two cases that
        # genuinely repeat inside a short span: a validation retry (same
        # analysis, same master prompt, within a minute or two) and a
        # multi-ticker session (this same master prompt reused across
        # several tickers analyzed back to back, e.g. the SKIP-gate batch).
        system=[
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ],
        thinking=thinking,
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )

    report_text = "".join(block.text for block in response.content if block.type == "text")

    if response.stop_reason == "max_tokens":
        raise TruncatedResponseError(cost_inr, len(report_text), call_max_tokens)

    verdict = extract_verdict_json(report_text)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_inr": cost_inr,
        "stage2_mode": mode,
        "model": call_model,
        "thinking_enabled": thinking is not None,
    }

    return report_text, verdict, usage
