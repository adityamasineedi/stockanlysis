"""Module 10 — deterministic validation. Pure Python, no LLM.

Reads verdict_json via llm.verdict.extract_verdict_json — never regexes
the prose to pull out values, since that was the brittler v1 design and
this is the layer everything downstream depends on. The one exception
(pledge check) searches the prose for the *absence* of a claim, which is
a fundamentally safer kind of check than extracting-and-comparing a value
out of free text: a false positive here just means an unnecessary retry,
never a wrong number silently accepted.

Also enforces the v3 master-prompt deployment checklist on every Stage 2
report: citation IDs, known placeholder tokens, §11 bear-downside PASS
line for >40x names, output order (Beginner Summary → JSON → Footer),
empty/thin-context guards.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from stockbot.analysis_routing import Stage2Mode
from stockbot.constitution_gates import apply_constitution_overrides, should_anti_chase
from stockbot.trade_policy import five_year_allows_buy_zone, wc_gap_blocks_buy_zone
from stockbot.expected_return import report_contains_yearly_return_ladder
from stockbot.llm.verdict import (
    ValuationComputed,
    VerdictJSON,
    VerdictParseError,
    compute_valuation,
    extract_verdict_json,
)
from stockbot.models import Brief, ValidationResult

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

STALENESS_TRADING_DAYS = 5

# Bear-case calibration thresholds (17A/17B). Found live on a real VMM
# report: the model's "bear" case assumed +11% EPS growth (₹2.00 against a
# real trailing EPS of ₹1.91) on a 58x stock — a bear case that assumes
# growth is a mild base case, not a bear case, and systematically
# understates the single most important number for a buy decision.
HIGH_MULTIPLE_PE_THRESHOLD = 40.0
MIN_BEAR_DOWNSIDE_PCT_ABOVE_HIGH_MULTIPLE = 30.0

# "X% below base fair value midpoint" per the master prompt's §13. Interpreted
# here as: the discount spanned by buy_zone_abs (top to bottom) relative to
# the fair_value_abs midpoint should fall within the risk-appropriate band,
# with a few points of tolerance since the master prompt itself says "no
# false precision" — this is a code-side backstop on a human-legible number,
# not a bit-exact contract.
RISK_DISCOUNT_BANDS = {
    "LOW": (0.10, 0.15),
    "MEDIUM": (0.20, 0.25),
    "HIGH": (0.35, 1.00),
}
# Found live on a real KPITTECH report: a MEDIUM-risk buy zone whose top was
# only 18.6% below fair-value midpoint (band floor 20%, so 1.4 points short)
# passed anyway, because the old 0.03 (3-point) tolerance treated anything
# down to 17% as "close enough". That swallows a real, visible band miss —
# tightened so a violation this size is caught instead of tolerated.
DISCOUNT_TOLERANCE = 0.01

# Found live on a real KPITTECH report: §1 wrote "Confidence: 5/7" — the
# pipeline cap is 7 on a 10-point scale, not a /7 scale. Master prompt
# explicitly: "never X/7". Catch the prose form so a retry fixes it.
_CONFIDENCE_OVER_SEVEN_RE = re.compile(
    r"\bconfidence\b[^.\n]{0,40}?\b(\d{1,2})\s*/\s*7\b",
    re.IGNORECASE,
)

# Found live on the same KPITTECH report: tokens wrapped in backticks
# (`` `{{buy_zone_low}}`–`{{buy_zone_high}}` ``) rendered as
# `` `₹400.00`–`₹430.00` `` — literal backticks stuck to every money figure
# in the Quick Verdict line. Master prompt forbids wrapping tokens in backticks.
_BACKTICK_RUPEE_RE = re.compile(r"`\s*₹\s*[\d,]+(?:\.\d+)?\s*`")

# Headline Fair Value must be the BASE range, never bear-low–bull-high.
_HEADLINE_FAIR_VALUE_RE = re.compile(
    r"fair\s*value[^₹\n]{0,40}?₹\s*([\d,]+(?:\.\d+)?)\s*[–\-]\s*₹\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

_PLEDGE_PCT_RE = re.compile(r"pledg\w*[^.\n]{0,50}?\d+(?:\.\d+)?\s*%", re.IGNORECASE)

# The master prompt's own §16 ("WHAT WOULD CHANGE THE VERDICT") requires
# stating hypothetical trigger thresholds, e.g. "Downgrade to SKIP: promoter
# pledge confirmed above 40%" — this is a forward-looking condition, not a
# claim about the current pledge level, and every well-formed report will
# contain one. Found live: this was flagging real, correctly-unconfirmed
# reports as having "invented" a pledge percentage, when the model had
# actually written "unconfirmed" everywhere and this was the only percentage
# in the whole report. Excise that section before checking for a stated
# current pledge figure.
_HYPOTHETICAL_SECTION_RE = re.compile(
    r"#*\s*16\.?\s*what would change the verdict.*?(?=\n#|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Technical-figure consistency: same "search for a specific numeric claim
# in prose" exception as the pledge check above, not a general-purpose
# value extractor — narrowly scoped to the exact figures Stage 2 is told
# are [FACT] and must not recompute.
# The period length ("14") commonly appears right next to the label itself
# (e.g. "RSI(14): 59.82", "RSI-14") — skip that label before capturing the
# actual value, or the regex grabs "14" as if it were the reading. Found
# live: a correctly-restated RSI of 59.82 was flagged as a mismatch because
# "RSI(14)" matched first, capturing "14" instead of "59.82".
# \b is required: without it, "rsi" matches as a bare substring inside any
# word containing those three letters — "reveRSIon", "diveRSIfication" —
# and .search() returns the FIRST match in the whole report, so a false hit
# earlier in the document (e.g. "mean reversion toward 24-26%") pre-empts
# the real, correct RSI statement later on. Found live on a real Opus call:
# this burned one full wasted validation retry (real money) on a report
# that had correctly stated RSI(14) 54.71 — the word "reversion" earlier in
# the same report was what actually matched.
#
# (?>...) atomic group (Python 3.11+): without it, "RSI14" with no real
# value left in the sentence (the real value is inside a stripped {{rsi14}}
# token) makes the engine backtrack into treating the label's own "14" as
# the captured value once it can't find a real one — found live on a real
# Sonnet 5 v3 report. Atomic grouping forbids that backtrack: once "14" is
# consumed as the label, it's never un-consumed to serve as the value.
_RSI_RE = re.compile(r"\brsi(?>[\s\-]*(?:\(?14\)?)?)\D{0,15}?(\d+(?:\.\d+)?)", re.IGNORECASE)
# Natural phrasing puts the period before the label ("50 DMA", "50-day
# SMA") at least as often as after it ("SMA50") — match either order.
_SMA50_RE = re.compile(
    r"(?:50[\s-]*(?:day)?[\s-]*(?:sma|dma)|(?:sma|dma)[\s-]*50)\D{0,15}?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SMA200_RE = re.compile(
    r"(?:200[\s-]*(?:day)?[\s-]*(?:sma|dma)|(?:sma|dma)[\s-]*200)\D{0,15}?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
TECHNICAL_FIGURE_TOLERANCE = 0.5

# v3 migration: the model is supposed to write a placeholder token
# ({{rsi14}}, {{sma50}}, {{sma200}}) rather than a literal number for these
# figures. Found live: validate_report runs on the RAW, unrendered report —
# the token names themselves contain digits ("rsi14" -> "14", "sma50" ->
# "50"), so a correctly-token-using report like "RSI14 of {{rsi14}}" was
# flagged as stating a wrong value, because the check's own regex matched
# the "14" inside the token's NAME, not an actual stated number. Strip
# placeholder tokens before scanning for literal figures — a model that
# uses the token correctly leaves nothing for these regexes to find (which
# is correct: nothing to check, since rendering guarantees the real value),
# while a model that ignores the token mandate and writes a literal wrong
# number is still caught.
_PLACEHOLDER_TOKEN_RE = re.compile(r"\{\{.*?\}\}")

# Deployment checklist (master prompt v3): valid [BRACKET] tags are source
# citation IDs plus evidence labels. [FACT]/[ANALYSIS]/etc. are labels,
# not source IDs, but they use the same bracket form and must not be
# rejected as "invalid citations".
VALID_BRACKET_TAGS: frozenset[str] = frozenset(
    {
        "PRICE_AND_TECHNICALS",
        "FINANCIALS",
        "SHAREHOLDING",
        "EXTRACTION",
        "MISSING",
        "PIPELINE_NOTE",
        "FACT",
        "ANALYSIS",
        "ESTIMATE",
        "UNVERIFIED",
    }
)
_BRACKET_TAG_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
_TOKEN_NAME_RE = re.compile(r"\{\{(\w+)\}\}")
_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BEAR_DOWNSIDE_LINE_RE = re.compile(
    r"Bear\s+downside\s+check(?:\s*\(revised\))?\s*:.*",
    re.IGNORECASE,
)
_FOOTER_NEEDLE = "Research and education, not investment advice"
_BEGINNER_SUMMARY_NEEDLE = "SHOULD I BUY?"

# Checks fixable in Python without a full Stage 2 regeneration.
AUTO_FIXABLE_CHECKS: frozenset[str] = frozenset(
    {
        "confidence_scale_over_ten",
        "no_backtick_wrapped_rupees",
    }
)

# Constitution gates the model should set but Python can patch deterministically.
CONSTITUTION_AUTO_FIX_CHECKS: frozenset[str] = frozenset(
    {
        "anti_chase_flag",
        "anti_chase_buy_block",
    }
)

# Prose-only fixes — narrow retry prompt instead of regenerating all sections.
NARROW_RETRY_CHECKS: frozenset[str] = frozenset(
    {
        "bear_downside_check_prose",
        "output_order",
        "citation_ids_valid",
        "standalone_disclosed",
        "headline_fair_value_is_base",
        "placeholder_tokens_known",
    }
)

class CheckResult(BaseModel):
    name: str
    passed: bool
    message: str


def _check_citation_ids_valid(report_text: str) -> CheckResult:
    found = _BRACKET_TAG_RE.findall(report_text)
    invalid = sorted({tag for tag in found if tag not in VALID_BRACKET_TAGS})
    if invalid:
        return CheckResult(
            name="citation_ids_valid",
            passed=False,
            message=f"invalid bracket tags (must be UPPERCASE from the allowed set): {invalid}",
        )
    return CheckResult(name="citation_ids_valid", passed=True, message="ok")


def _check_placeholder_tokens_known(report_text: str) -> CheckResult:
    from stockbot.render import ALLOWED_PLACEHOLDER_TOKENS

    found = _TOKEN_NAME_RE.findall(report_text)
    unknown = sorted({name for name in found if name not in ALLOWED_PLACEHOLDER_TOKENS})
    if unknown:
        return CheckResult(
            name="placeholder_tokens_known",
            passed=False,
            message=f"unknown placeholder token(s): {unknown}",
        )
    return CheckResult(name="placeholder_tokens_known", passed=True, message="ok")


def _section_eleven(report_text: str) -> str:
    """Return §11 body text when headings are present; else the full report."""
    match = re.search(
        r"#{0,3}\s*11\.\s*VALUATION\b(.*?)(?=\n#{0,3}\s*12\.|\n#\s*OUTPUT|\n\*\*SHOULD I BUY\?\*\*|\n```json|\Z)",
        report_text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    return report_text


def _check_bear_downside_check_prose(
    report_text: str, verdict: VerdictJSON, brief: Brief
) -> CheckResult:
    """Master prompt rule #5: for >40x trailing, §11 must end on a PASS line."""
    trailing_eps = _trailing_eps(brief)
    if trailing_eps is None or trailing_eps <= 0:
        return CheckResult(
            name="bear_downside_check_prose",
            passed=True,
            message="trailing EPS unavailable, not applicable",
        )
    current_pe = verdict.current_price_abs / trailing_eps
    if current_pe <= HIGH_MULTIPLE_PE_THRESHOLD:
        return CheckResult(
            name="bear_downside_check_prose",
            passed=True,
            message=f"not applicable ({current_pe:.0f}x <= 40x)",
        )

    section = _section_eleven(report_text)
    lines = _BEAR_DOWNSIDE_LINE_RE.findall(section)
    if not lines:
        return CheckResult(
            name="bear_downside_check_prose",
            passed=False,
            message=(
                f"{current_pe:.0f}x stock requires an explicit "
                f"'Bear downside check: … — PASS — …' line in §11"
            ),
        )
    last = lines[-1]
    if re.search(r"—\s*PASS\s*—", last, re.IGNORECASE):
        return CheckResult(name="bear_downside_check_prose", passed=True, message="ok")
    if re.search(r"—\s*FAIL\s*—", last, re.IGNORECASE):
        return CheckResult(
            name="bear_downside_check_prose",
            passed=False,
            message=(
                "§11 bear downside check ends on FAIL — must revise and leave a "
                "final 'Bear downside check (revised): … — PASS — …' line"
            ),
        )
    return CheckResult(
        name="bear_downside_check_prose",
        passed=False,
        message=f"§11 bear downside check line missing PASS/FAIL marker: {last[:120]!r}",
    )


def _check_output_order(report_text: str) -> CheckResult:
    """Enforce §1–16 → Beginner Summary → JSON → Footer (when structure is present)."""
    fences = list(_JSON_FENCE_RE.finditer(report_text))
    if not fences:
        return CheckResult(
            name="output_order",
            passed=False,
            message="no fenced json block found",
        )
    if len(fences) > 1:
        return CheckResult(
            name="output_order",
            passed=False,
            message=f"expected exactly one fenced json block, found {len(fences)}",
        )
    fence = fences[0]
    before = report_text[: fence.start()]
    after = report_text[fence.end() :]

    if _BEGINNER_SUMMARY_NEEDLE not in before:
        return CheckResult(
            name="output_order",
            passed=False,
            message="Beginner Summary ('SHOULD I BUY?') must appear before the JSON block",
        )
    if _FOOTER_NEEDLE.lower() not in after.lower():
        return CheckResult(
            name="output_order",
            passed=False,
            message="SEBI footer must appear after the JSON block (last in the response)",
        )

    has_s1 = bool(re.search(r"#{0,3}\s*1\.\s*QUICK VERDICT\b", before, re.IGNORECASE))
    has_s16 = bool(
        re.search(r"#{0,3}\s*16\.\s*WHAT WOULD CHANGE THE VERDICT", before, re.IGNORECASE)
    )
    if has_s1 and has_s16:
        s16 = re.search(
            r"#{0,3}\s*16\.\s*WHAT WOULD CHANGE THE VERDICT", before, re.IGNORECASE
        )
        beginner = before.rfind(_BEGINNER_SUMMARY_NEEDLE)
        if s16 is not None and beginner < s16.start():
            return CheckResult(
                name="output_order",
                passed=False,
                message="Beginner Summary must appear after §16, before JSON",
            )

    return CheckResult(name="output_order", passed=True, message="ok")


def _check_thin_context_no_invented_business(
    report_text: str, brief: Brief
) -> CheckResult:
    """Famous-company + thin-context guard: §2 must not invent a business model."""
    business_missing = brief.financials is None or (
        brief.financials.business_description is None
        or not str(brief.financials.business_description).strip()
    )
    # Also treat an explicit missing marker on the brief.
    explicit = any(
        "business" in item.lower() and "description" in item.lower()
        for item in brief.missing
    )
    if not business_missing and not explicit:
        return CheckResult(
            name="thin_context_business_model",
            passed=True,
            message="business description present in brief, not applicable",
        )

    section2 = re.search(
        r"#{0,3}\s*2\.\s*COMPANY IN 60 SECONDS\b(.*?)(?=\n#{0,3}\s*3\.|\n\*\*SHOULD I BUY\?\*\*|\n```json|\Z)",
        report_text,
        re.IGNORECASE | re.DOTALL,
    )
    if section2 is None:
        # Section absent — other structure checks may catch it; don't double-fail.
        return CheckResult(
            name="thin_context_business_model",
            passed=True,
            message="§2 not present, skipped",
        )
    body = section2.group(1)
    cites_missing = "[MISSING]" in body
    admits_gap = bool(
        re.search(
            r"cannot be determined|not available|MISSING|no (?:business|product|company) description",
            body,
            re.IGNORECASE,
        )
    )
    if cites_missing or admits_gap:
        return CheckResult(name="thin_context_business_model", passed=True, message="ok")
    return CheckResult(
        name="thin_context_business_model",
        passed=False,
        message=(
            "business description is missing from the brief but §2 does not cite "
            "[MISSING] or state the gap cannot be determined — likely filled from memory"
        ),
    )


def _check_empty_context_verdict(verdict: VerdictJSON, brief: Brief) -> CheckResult:
    """Empty/near-empty evidence: must not produce a BUY and must keep confidence low."""
    if len(brief.missing) < 4 and brief.financials is not None:
        return CheckResult(
            name="empty_context_verdict",
            passed=True,
            message="context not empty, not applicable",
        )
    # Heavy MISSING and no financials → SKIP/WATCH only, confidence ≤ 2 when
    # essentially everything is gone (≥6 missing or financials None + shareholding None).
    skeletal = brief.financials is None and brief.shareholding is None
    heavy = len(brief.missing) >= 6 or skeletal
    if not heavy:
        return CheckResult(
            name="empty_context_verdict",
            passed=True,
            message="not applicable",
        )
    if verdict.verdict == "BUY":
        return CheckResult(
            name="empty_context_verdict",
            passed=False,
            message=f"empty/thin context must not yield BUY (got {verdict.verdict!r})",
        )
    if skeletal and verdict.confidence > 2:
        return CheckResult(
            name="empty_context_verdict",
            passed=False,
            message=(
                f"skeletal context (no financials, no shareholding) requires "
                f"confidence≤2, got {verdict.confidence}"
            ),
        )
    return CheckResult(name="empty_context_verdict", passed=True, message="ok")


def _check_confidence_cap(verdict: VerdictJSON, brief: Brief) -> CheckResult:
    # Pipeline hard max is 7; brief.confidence_ceiling can be lower when
    # financials/annual-report fetches failed (see brief.assemble_brief).
    cap = min(7, brief.confidence_ceiling)
    return CheckResult(
        name="confidence_cap",
        passed=verdict.confidence <= cap,
        message=f"confidence={verdict.confidence} (cap={cap})",
    )


def _check_buy_gate(verdict: VerdictJSON) -> CheckResult:
    gate_triggered = (
        verdict.business_quality < 7
        or verdict.management_quality < 7
        or verdict.earnings_quality == "LOW"
    )
    violated = gate_triggered and verdict.verdict == "BUY"
    return CheckResult(
        name="verdict_gate",
        passed=not violated,
        message=(
            f"business_quality={verdict.business_quality}, "
            f"management_quality={verdict.management_quality}, "
            f"earnings_quality={verdict.earnings_quality!r}, verdict={verdict.verdict!r}"
        ),
    )


def _check_ranges_ordered(verdict: VerdictJSON, valuation: ValuationComputed) -> CheckResult:
    # fair_value_*_abs are Python-computed (see compute_valuation) and
    # already sorted defensively for negative EPS, so this is really
    # checking buy_zone_abs (still model-stated) plus a sanity confirmation
    # the computed ranges came out sane. buy_zone_abs may be null when
    # constitution gates block a range.
    ranges: dict[str, tuple[float, float]] = {
        "fair_value_base_abs": valuation.fair_value_base_abs,
        "fair_value_bear_abs": valuation.fair_value_bear_abs,
        "fair_value_bull_abs": valuation.fair_value_bull_abs,
    }
    if verdict.buy_zone_abs is not None:
        ranges["buy_zone_abs"] = verdict.buy_zone_abs
    bad = [name for name, (low, high) in ranges.items() if not (low < high)]
    return CheckResult(
        name="ranges_ordered",
        passed=not bad,
        message=f"out-of-order ranges: {bad}" if bad else "ok",
    )


def _check_buy_zone_discount(verdict: VerdictJSON, valuation: ValuationComputed) -> CheckResult:
    if verdict.buy_zone_abs is None or verdict.buy_range_allowed is False:
        return CheckResult(
            name="buy_zone_discount",
            passed=True,
            message="buy zone not issued — discount check skipped",
        )
    band = RISK_DISCOUNT_BANDS.get(verdict.risk)
    if band is None:
        return CheckResult(
            name="buy_zone_discount", passed=False, message=f"unknown risk level {verdict.risk!r}"
        )
    band_min, band_max = band

    fv_mid = (valuation.fair_value_base_abs[0] + valuation.fair_value_base_abs[1]) / 2
    if fv_mid <= 0:
        return CheckResult(
            name="buy_zone_discount", passed=False, message="fair value midpoint is not positive"
        )

    buy_low, buy_high = verdict.buy_zone_abs
    discount_at_top = (fv_mid - buy_high) / fv_mid
    discount_at_bottom = (fv_mid - buy_low) / fv_mid

    if discount_at_top < band_min - DISCOUNT_TOLERANCE:
        return CheckResult(
            name="buy_zone_discount",
            passed=False,
            message=(
                f"buy zone top {buy_high} is only {discount_at_top:.1%} below fair value "
                f"midpoint {fv_mid:.1f}, below the {verdict.risk} band "
                f"({band_min:.0%}-{band_max:.0%})"
            ),
        )
    if discount_at_bottom > band_max + DISCOUNT_TOLERANCE:
        return CheckResult(
            name="buy_zone_discount",
            passed=False,
            message=(
                f"buy zone bottom {buy_low} is {discount_at_bottom:.1%} below fair value "
                f"midpoint {fv_mid:.1f}, beyond the {verdict.risk} band "
                f"({band_min:.0%}-{band_max:.0%})"
            ),
        )
    return CheckResult(name="buy_zone_discount", passed=True, message="within band")


def _check_five_year_buy_gate(verdict: VerdictJSON) -> CheckResult:
    """Constitution: no buy/add range unless five-year test is YES."""
    test = verdict.five_year_business_test
    if test is None:
        return CheckResult(
            name="five_year_buy_gate",
            passed=True,
            message="five_year_business_test omitted — gate not enforced on legacy JSON",
        )
    answer = (test.answer or "").strip().upper()
    if five_year_allows_buy_zone(test):
        label = answer or "YES"
        return CheckResult(name="five_year_buy_gate", passed=True, message=label)

    blocked_verdicts = {"BUY", "BUY ON CORRECTION"}
    range_claimed = verdict.buy_range_allowed is True or verdict.add_range_allowed is True
    zone_present = verdict.buy_zone_abs is not None
    verdict_blocked = verdict.verdict.strip().upper() in blocked_verdicts
    violated = range_claimed or zone_present or verdict_blocked
    return CheckResult(
        name="five_year_buy_gate",
        passed=not violated,
        message=(
            f"five_year_answer={answer!r}, buy_range_allowed={verdict.buy_range_allowed!r}, "
            f"add_range_allowed={verdict.add_range_allowed!r}, "
            f"buy_zone_abs={'set' if zone_present else 'null'}, verdict={verdict.verdict!r}"
        ),
    )


def _financial_years_available(brief: Brief) -> int:
    if brief.financials is None:
        return 0
    pnl = brief.financials.pnl
    if pnl is None or pnl.empty:
        return 0
    return len(pnl.columns)


def _check_five_year_uncertain_requires_evidence(
    verdict: VerdictJSON, brief: Brief
) -> CheckResult:
    """Block lazy UNCERTAIN when pipeline data is complete — forces named trends."""
    test = verdict.five_year_business_test
    if test is None:
        return CheckResult(
            name="five_year_uncertain_evidence",
            passed=True,
            message="five_year omitted",
        )
    answer = (test.answer or "").strip().upper()
    if answer != "UNCERTAIN":
        return CheckResult(
            name="five_year_uncertain_evidence",
            passed=True,
            message=f"answer={answer!r}",
        )
    if brief.missing:
        return CheckResult(
            name="five_year_uncertain_evidence",
            passed=True,
            message="pipeline gaps present",
        )
    if _financial_years_available(brief) < 5:
        return CheckResult(
            name="five_year_uncertain_evidence",
            passed=True,
            message="thin financials",
        )
    evidence_against = [str(item).strip() for item in (test.evidence_against or []) if str(item).strip()]
    evidence_for = [str(item).strip() for item in (test.evidence_for or []) if str(item).strip()]
    if evidence_against or evidence_for:
        return CheckResult(
            name="five_year_uncertain_evidence",
            passed=True,
            message="evidence listed",
        )
    return CheckResult(
        name="five_year_uncertain_evidence",
        passed=False,
        message=(
            "five_year UNCERTAIN with complete FINANCIALS but empty evidence_for/against — "
            "cite specific multi-year trends from FINANCIALS or answer YES/NO"
        ),
    )


_OCF_ROW_ALIASES = (
    "Cash from Operating Activity",
    "Cash from Operating Activities",
)
_PAT_ROW_ALIASES = ("Net Profit", "Profit after Tax", "PAT")
_WC_UNLOCK_CLASSIFICATION = "TEMPORARY_BILLING_CYCLE"
_WC_BLOCK_CLASSIFICATIONS = frozenset(
    {
        "WORKING_CAPITAL_STRESS",
        "DATA_OR_SCOPE_ERROR",
        "INCONCLUSIVE",
    }
)
_ESCALATED_CUM_OCF_PAT = 0.25


def _numeric_series_from_statement(
    frame: pd.DataFrame, aliases: tuple[str, ...]
) -> list[float]:
    for name in aliases:
        if name not in frame.index:
            continue
        row = frame.loc[name]
        values: list[float] = []
        for col in row.index:
            if str(col).strip().upper() == "TTM":
                continue
            raw = row[col]
            if pd.isna(raw):
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if values:
            return values
    return []


def brief_shows_extreme_cash_conversion(brief: Brief) -> bool:
    """True when supplied FINANCIALS show Mazdock-style reported OCF weakness."""
    if brief.financials is None:
        return False
    ocf = _numeric_series_from_statement(brief.financials.cash_flow, _OCF_ROW_ALIASES)
    pat = _numeric_series_from_statement(brief.financials.pnl, _PAT_ROW_ALIASES)
    if not ocf or not pat:
        return False

    n = min(len(ocf), len(pat), 3)
    if n >= 2:
        ocf_sum = sum(ocf[-n:])
        pat_sum = sum(pat[-n:])
        if pat_sum > 0 and (ocf_sum / pat_sum) < _ESCALATED_CUM_OCF_PAT:
            return True

    # Sharply negative latest OCF against positive latest PAT
    return ocf[-1] < 0 and pat[-1] > 0 and abs(ocf[-1]) > 0.25 * pat[-1]


def _claims_buy_or_add_range(verdict: VerdictJSON) -> bool:
    if verdict.buy_range_allowed is True or verdict.add_range_allowed is True:
        return True
    if verdict.buy_zone_abs is not None:
        return True
    return verdict.verdict.strip().upper() in {"BUY", "BUY ON CORRECTION"}


def _check_wc_buy_gate(verdict: VerdictJSON, brief: Brief) -> CheckResult:
    """No buy/add range until WC gap is TEMPORARY_BILLING_CYCLE when cash is extreme."""
    classification_raw = verdict.wc_gap_classification
    classification = (
        classification_raw.strip().upper() if isinstance(classification_raw, str) else None
    )
    if classification == "":
        classification = None

    claiming = _claims_buy_or_add_range(verdict)
    extreme = brief_shows_extreme_cash_conversion(brief)

    if classification is not None and claiming and wc_gap_blocks_buy_zone(classification):
        return CheckResult(
            name="wc_buy_gate",
            passed=False,
            message=(
                f"wc_gap_classification={classification!r} blocks buy/add ranges; "
                f"only {_WC_UNLOCK_CLASSIFICATION} unlocks them"
            ),
        )

    if extreme and claiming and classification != _WC_UNLOCK_CLASSIFICATION:
        if classification is None or wc_gap_blocks_buy_zone(classification):
            return CheckResult(
                name="wc_buy_gate",
                passed=False,
                message=(
                    "Reported cash conversion is extremely weak in FINANCIALS; "
                    f"set wc_gap_classification={_WC_UNLOCK_CLASSIFICATION} with "
                    "year-by-year CFO-to-PAT evidence, or set buy_zone_abs=null and "
                    "buy_range_allowed=false"
                ),
            )

    return CheckResult(
        name="wc_buy_gate",
        passed=True,
        message=(
            f"ok (extreme={extreme}, classification={classification!r}, claiming={claiming})"
        ),
    )


def _check_anti_chase_flag(
    verdict: VerdictJSON, valuation: ValuationComputed, brief: Brief
) -> CheckResult:
    expected, reason = should_anti_chase(verdict, valuation, brief)
    if not expected:
        return CheckResult(name="anti_chase_flag", passed=True, message="not required")
    if verdict.anti_chase_flag is True:
        return CheckResult(name="anti_chase_flag", passed=True, message=reason)
    return CheckResult(
        name="anti_chase_flag",
        passed=False,
        message=f"anti_chase_flag must be true — {reason}",
    )


def _check_anti_chase_blocks_buy_range(verdict: VerdictJSON) -> CheckResult:
    if verdict.anti_chase_flag is not True:
        return CheckResult(name="anti_chase_buy_block", passed=True, message="ok")
    claiming = (
        verdict.buy_range_allowed is True
        or verdict.add_range_allowed is True
        or verdict.buy_zone_abs is not None
    )
    return CheckResult(
        name="anti_chase_buy_block",
        passed=not claiming,
        message=(
            "anti_chase_flag=true requires buy_range_allowed=false and buy_zone_abs=null"
            if claiming
            else "ok"
        ),
    )


def _check_holding_period_vs_thesis(verdict: VerdictJSON) -> CheckResult:
    """Thesis under review → monitoring horizon, not committed 3–5y holding."""
    status = (verdict.thesis_status or "").strip().upper()
    if status not in {"THESIS_UNDER_REVIEW", "THESIS_AT_RISK", "THESIS_BROKEN"}:
        return CheckResult(name="holding_period_vs_thesis", passed=True, message="ok")
    hp = (verdict.holding_period or "").lower()
    long_term_markers = ("3-5", "3–5", "5+", "5+ years", "five year", "5 year")
    if any(m in hp for m in long_term_markers):
        return CheckResult(
            name="holding_period_vs_thesis",
            passed=False,
            message=(
                f"thesis_status={status!r} with holding_period={verdict.holding_period!r} "
                "— use a monitoring horizon (e.g. 6–12 months) until thesis confirms"
            ),
        )
    return CheckResult(name="holding_period_vs_thesis", passed=True, message="ok")


def _check_confidence_vs_missing_data(verdict: VerdictJSON, brief: Brief) -> CheckResult:
    # Found live on the real BEL run: 7 MISSING items (including the order
    # book, which the report itself called "the single most important
    # number for this company") and the model still claimed confidence 7 —
    # the pipeline maximum. A lot of MISSING must visibly lower confidence;
    # this makes that non-optional instead of trusting the model to apply
    # v3's own "every MISSING item should visibly move this number down"
    # instruction unprompted.
    n_missing = len(brief.missing)
    if n_missing > 6:
        cap = 4
    elif n_missing > 4:
        cap = 5
    else:
        return CheckResult(name="confidence_vs_missing_data", passed=True, message="ok")
    return CheckResult(
        name="confidence_vs_missing_data",
        passed=verdict.confidence <= cap,
        message=(
            f"{n_missing} MISSING items requires confidence<={cap}, got {verdict.confidence}"
            if verdict.confidence > cap
            else "ok"
        ),
    )


def _trailing_eps(brief: Brief) -> float | None:
    # Screener's P&L table reliably labels this row "EPS in Rs", with a
    # "TTM" column when quarterly data is available — real, checked
    # against VMM's actual fetched data (FY26 1.80, TTM 1.91, matching the
    # real report exactly). Falls back to the most recent FY column only
    # when TTM genuinely isn't present, never guesses a value.
    if brief.financials is None:
        return None
    pnl = brief.financials.pnl
    if "EPS in Rs" not in pnl.index:
        return None
    row = pnl.loc["EPS in Rs"]
    value = row["TTM"] if "TTM" in pnl.columns else row.iloc[-1]
    return float(value) if pd.notna(value) else None


def _check_bear_eps_sanity(verdict: VerdictJSON, brief: Brief) -> CheckResult:
    trailing_eps = _trailing_eps(brief)
    if trailing_eps is None:
        return CheckResult(name="bear_eps_sanity", passed=True, message="trailing EPS unavailable, not applicable")

    eps_bear = verdict.valuation_inputs.eps_bear
    if eps_bear <= trailing_eps:
        return CheckResult(name="bear_eps_sanity", passed=True, message="ok")
    if verdict.bear_growth_justification:
        return CheckResult(
            name="bear_eps_sanity",
            passed=True,
            message=f"bear EPS {eps_bear} exceeds TTM {trailing_eps} but justified",
        )
    return CheckResult(
        name="bear_eps_sanity",
        passed=False,
        message=(
            f"bear EPS {eps_bear} exceeds trailing EPS {trailing_eps} with no "
            f"bear_growth_justification — a bear case assumes growth stops, not that it slows"
        ),
    )


def _check_bear_adequacy_for_high_multiple(
    verdict: VerdictJSON, valuation: ValuationComputed, brief: Brief
) -> CheckResult:
    trailing_eps = _trailing_eps(brief)
    if trailing_eps is None or trailing_eps <= 0:
        return CheckResult(
            name="bear_adequacy_high_multiple", passed=True, message="trailing EPS unavailable, not applicable"
        )

    current_pe = verdict.current_price_abs / trailing_eps
    if current_pe <= HIGH_MULTIPLE_PE_THRESHOLD:
        return CheckResult(
            name="bear_adequacy_high_multiple", passed=True, message=f"not applicable ({current_pe:.0f}x <= 40x)"
        )

    fair_value_bear_mid = (valuation.fair_value_bear_abs[0] + valuation.fair_value_bear_abs[1]) / 2
    # Deliberately NOT the same convention as render.py's {{downside_pct}}
    # token (which is signed: (bear_mid - current) / current, negative when
    # bear sits below current). This is a local threshold check, not a
    # display value — kept as an unsigned magnitude ("how adverse is the
    # bear case, as a % of current price") since MIN_BEAR_DOWNSIDE_PCT_...
    # reads naturally as "at least 30% downside", not "at most -30%". Do
    # not substitute render.py's token value in here; the sign differs.
    downside_pct = (verdict.current_price_abs - fair_value_bear_mid) / verdict.current_price_abs * 100
    if downside_pct >= MIN_BEAR_DOWNSIDE_PCT_ABOVE_HIGH_MULTIPLE:
        return CheckResult(
            name="bear_adequacy_high_multiple", passed=True, message=f"ok, downside {downside_pct:.1f}%"
        )
    return CheckResult(
        name="bear_adequacy_high_multiple",
        passed=False,
        message=(
            f"bear case insufficiently adverse for a {current_pe:.0f}x stock: only "
            f"{downside_pct:.1f}% downside, need >={MIN_BEAR_DOWNSIDE_PCT_ABOVE_HIGH_MULTIPLE:.0f}%"
        ),
    )


def _check_price_date_fresh(verdict: VerdictJSON) -> CheckResult:
    today = datetime.now(UTC).date()
    trading_days_elapsed = int(np.busday_count(verdict.price_date, today))
    return CheckResult(
        name="price_date_fresh",
        passed=trading_days_elapsed <= STALENESS_TRADING_DAYS,
        message=f"price_date={verdict.price_date.isoformat()}, {trading_days_elapsed} trading days old",
    )


def _check_pledge_not_stated_when_unconfirmed(report_text: str, brief: Brief) -> CheckResult:
    pledge_confirmed = (
        brief.shareholding is not None
        and brief.shareholding.pledge_pct_of_promoter_holding is not None
    )
    if pledge_confirmed:
        return CheckResult(name="pledge_not_invented", passed=True, message="pledge was confirmed")

    factual_text = _HYPOTHETICAL_SECTION_RE.sub("", report_text)
    stated = bool(_PLEDGE_PCT_RE.search(factual_text))
    return CheckResult(
        name="pledge_not_invented",
        passed=not stated,
        message=(
            "report states a pledge percentage that was never confirmed by an exchange source"
            if stated
            else "ok"
        ),
    )


def _check_standalone_disclosed(report_text: str, brief: Brief) -> CheckResult:
    if brief.financials is None or brief.financials.basis != "standalone":
        return CheckResult(name="standalone_disclosed", passed=True, message="not applicable")
    disclosed = "standalone" in report_text.lower()
    return CheckResult(
        name="standalone_disclosed",
        passed=disclosed,
        message="ok" if disclosed else "financials are standalone but the report never says so",
    )


def _check_technical_figures_not_recomputed(report_text: str, brief: Brief) -> CheckResult:
    # Hard injection #1 tells Stage 2 the technicals are [FACT], computed
    # in code, and must not be recomputed or estimated. This checks the
    # model actually honoured that: if the report states an RSI/SMA figure
    # that clearly differs from what was provided, it recomputed/guessed
    # instead of using the given value. Only flags a figure that's both
    # mentioned AND wrong — absence isn't an error (the model may simply
    # not restate every number in prose).
    text_without_tokens = _PLACEHOLDER_TOKEN_RE.sub("", report_text)
    mismatches: list[str] = []
    checks = [
        ("RSI14", _RSI_RE, brief.technicals.rsi14),
        ("SMA50", _SMA50_RE, brief.technicals.sma50),
        ("SMA200", _SMA200_RE, brief.technicals.sma200),
    ]
    for name, pattern, expected in checks:
        if expected is None:
            continue
        match = pattern.search(text_without_tokens)
        if match is None:
            continue
        stated = float(match.group(1))
        if abs(stated - expected) > TECHNICAL_FIGURE_TOLERANCE:
            mismatches.append(f"{name}: report states {stated}, computed value was {expected:.2f}")

    return CheckResult(
        name="technical_figures_not_recomputed",
        passed=not mismatches,
        message="; ".join(mismatches) if mismatches else "ok",
    )


def _parse_rupee(raw: str) -> float:
    return float(raw.replace(",", ""))


def _check_confidence_scale_is_over_ten(report_text: str) -> CheckResult:
    match = _CONFIDENCE_OVER_SEVEN_RE.search(report_text)
    if match is None:
        return CheckResult(name="confidence_scale_over_ten", passed=True, message="ok")
    return CheckResult(
        name="confidence_scale_over_ten",
        passed=False,
        message=(
            f"report states Confidence {match.group(1)}/7 — scale is always X/10 "
            f"(7 is the pipeline cap, not the denominator). Rewrite as {match.group(1)}/10."
        ),
    )


def _check_no_backtick_wrapped_rupees(report_text: str) -> CheckResult:
    hits = _BACKTICK_RUPEE_RE.findall(report_text)
    if not hits:
        return CheckResult(name="no_backtick_wrapped_rupees", passed=True, message="ok")
    return CheckResult(
        name="no_backtick_wrapped_rupees",
        passed=False,
        message=(
            f"found {len(hits)} backtick-wrapped ₹ amounts (e.g. {hits[0]}) — "
            "write tokens bare ({{buy_zone_low}} not `{{buy_zone_low}}`); "
            "Python substitutes plain numbers"
        ),
    )


def _check_headline_fair_value_is_base(
    report_text: str, valuation: ValuationComputed
) -> CheckResult:
    """Reject bear-low–bull-high as the headline Fair Value span."""
    match = _HEADLINE_FAIR_VALUE_RE.search(report_text)
    if match is None:
        return CheckResult(name="headline_fair_value_is_base", passed=True, message="ok")

    low = _parse_rupee(match.group(1))
    high = _parse_rupee(match.group(2))
    bear_low, _bear_high = valuation.fair_value_bear_abs
    _bull_low, bull_high = valuation.fair_value_bull_abs
    base_low, base_high = valuation.fair_value_base_abs

    spans_bear_to_bull = abs(low - bear_low) <= 1.0 and abs(high - bull_high) <= 1.0
    matches_base = abs(low - base_low) <= 1.0 and abs(high - base_high) <= 1.0
    if spans_bear_to_bull and not matches_base:
        return CheckResult(
            name="headline_fair_value_is_base",
            passed=False,
            message=(
                f"headline Fair Value ₹{low:.2f}–₹{high:.2f} spans bear-low to bull-high; "
                f"must be the BASE range ₹{base_low:.2f}–₹{base_high:.2f} "
                f"(use {{{{fair_value_base_low}}}}–{{{{fair_value_base_high}}}})"
            ),
        )
    return CheckResult(name="headline_fair_value_is_base", passed=True, message="ok")


def _failed_check_names(result: ValidationResult) -> set[str]:
    names: set[str] = set()
    for failure in result.failures:
        if ":" in failure:
            names.add(failure.split(":", 1)[0].strip())
    return names


def classify_retry_mode(result: ValidationResult) -> Literal["narrow", "full"]:
    names = _failed_check_names(result)
    if names and names <= NARROW_RETRY_CHECKS:
        return "narrow"
    return "full"


def _patch_verdict_json_block(report_text: str, verdict: VerdictJSON) -> str:
    """Replace the last ```json verdict block with an updated payload."""
    matches = list(_JSON_BLOCK_RE.finditer(report_text))
    if not matches:
        return report_text
    last = matches[-1]
    new_inner = json.dumps(verdict.model_dump(mode="json"), indent=2)
    replacement = f"```json\n{new_inner}\n```"
    return report_text[: last.start()] + replacement + report_text[last.end() :]


def try_auto_fix_report(
    report_text: str, result: ValidationResult, brief: Brief, *, stage2_mode: Stage2Mode = "FULL"
) -> tuple[str, ValidationResult] | None:
    """Apply deterministic fixes when failures are purely formatting/constitution."""
    names = _failed_check_names(result)
    if not names:
        return None

    fixed = report_text
    validation = result

    if names <= CONSTITUTION_AUTO_FIX_CHECKS:
        try:
            verdict = extract_verdict_json(fixed)
        except VerdictParseError:
            return None
        valuation = compute_valuation(verdict.valuation_inputs)
        patched = apply_constitution_overrides(verdict, valuation, brief)
        fixed = _patch_verdict_json_block(fixed, patched)
        validation = validate_report(fixed, brief, stage2_mode=stage2_mode)
        if validation.passed:
            logger.info(
                "Auto-fixed constitution validation failures without Stage 2 retry: %s",
                sorted(names),
            )
            return fixed, validation
        names = _failed_check_names(validation)
        if not names:
            return fixed, validation

    if not names or not names <= AUTO_FIXABLE_CHECKS:
        return (fixed, validation) if validation.passed else None

    if "confidence_scale_over_ten" in names:

        def _fix_confidence_scale(match: re.Match[str]) -> str:
            return match.group(0).replace("/7", "/10")

        fixed = _CONFIDENCE_OVER_SEVEN_RE.sub(_fix_confidence_scale, fixed)
    if "no_backtick_wrapped_rupees" in names:
        fixed = _BACKTICK_RUPEE_RE.sub(lambda m: m.group(0).strip("`"), fixed)

    revalidated = validate_report(fixed, brief, stage2_mode=stage2_mode)
    if revalidated.passed:
        logger.info("Auto-fixed validation failures without Stage 2 retry: %s", sorted(names))
    return fixed, revalidated


def _check_no_yearly_return_ladder(report_text: str) -> CheckResult:
    if report_contains_yearly_return_ladder(report_text):
        return CheckResult(
            name="no_yearly_return_ladder",
            passed=False,
            message=(
                "report states fixed per-year return targets (e.g. 'year 1 = 12%') — "
                "use scenario CAGR ranges over 2–5 years instead, not a yearly ladder"
            ),
        )
    return CheckResult(name="no_yearly_return_ladder", passed=True, message="ok")


def validate_report(
    report_text: str, brief: Brief, *, stage2_mode: Stage2Mode = "FULL"
) -> ValidationResult:
    try:
        verdict = extract_verdict_json(report_text)
    except VerdictParseError as exc:
        return ValidationResult(passed=False, failures=[f"verdict_json_parse: {exc}"])

    valuation = compute_valuation(verdict.valuation_inputs)

    checks = [
        _check_confidence_cap(verdict, brief),
        _check_buy_gate(verdict),
        _check_five_year_buy_gate(verdict),
        _check_five_year_uncertain_requires_evidence(verdict, brief),
        _check_wc_buy_gate(verdict, brief),
        _check_anti_chase_flag(verdict, valuation, brief),
        _check_anti_chase_blocks_buy_range(verdict),
        _check_holding_period_vs_thesis(verdict),
        _check_ranges_ordered(verdict, valuation),
        _check_buy_zone_discount(verdict, valuation),
        _check_confidence_vs_missing_data(verdict, brief),
        _check_bear_eps_sanity(verdict, brief),
        _check_bear_adequacy_for_high_multiple(verdict, valuation, brief),
        _check_price_date_fresh(verdict),
        _check_pledge_not_stated_when_unconfirmed(report_text, brief),
        _check_technical_figures_not_recomputed(report_text, brief),
        _check_confidence_scale_is_over_ten(report_text),
        _check_no_backtick_wrapped_rupees(report_text),
        _check_no_yearly_return_ladder(report_text),
        _check_citation_ids_valid(report_text),
        _check_output_order(report_text),
    ]
    if stage2_mode == "FULL":
        checks.extend(
            [
                _check_bear_downside_check_prose(report_text, verdict, brief),
                _check_standalone_disclosed(report_text, brief),
                _check_headline_fair_value_is_base(report_text, valuation),
                _check_placeholder_tokens_known(report_text),
                _check_empty_context_verdict(verdict, brief),
                _check_thin_context_no_invented_business(report_text, brief),
            ]
        )
    else:
        checks.extend(
            [
                _check_headline_fair_value_is_base(report_text, valuation),
                _check_placeholder_tokens_known(report_text),
            ]
        )

    failures = [f"{c.name}: {c.message}" for c in checks if not c.passed]
    return ValidationResult(passed=not failures, failures=failures)


def format_validation_errors(
    result: ValidationResult, *, retry_mode: Literal["narrow", "full"] = "full"
) -> str:
    if retry_mode == "narrow":
        lines = [
            "The previous attempt failed these checks — fix ONLY the listed issues.",
            "Keep all other sections and the JSON block unchanged unless a fix requires a small edit:",
        ]
    else:
        lines = ["The previous attempt failed these checks — fix them and resend the FULL report:"]
    lines.extend(f"- {failure}" for failure in result.failures)
    return "\n".join(lines)
