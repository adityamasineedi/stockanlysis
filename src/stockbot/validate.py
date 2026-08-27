"""Module 10 — deterministic validation. Pure Python, no LLM.

Reads verdict_json via llm.verdict.extract_verdict_json — never regexes
the prose to pull out values, since that was the brittler v1 design and
this is the layer everything downstream depends on. The one exception
(pledge check) searches the prose for the *absence* of a claim, which is
a fundamentally safer kind of check than extracting-and-comparing a value
out of free text: a false positive here just means an unnecessary retry,
never a wrong number silently accepted.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel

from stockbot.llm.verdict import (
    ValuationComputed,
    VerdictJSON,
    VerdictParseError,
    compute_valuation,
    extract_verdict_json,
)
from stockbot.models import Brief, ValidationResult

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


class CheckResult(BaseModel):
    name: str
    passed: bool
    message: str


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
    # the computed ranges came out sane.
    ranges = {
        "buy_zone_abs": verdict.buy_zone_abs,
        "fair_value_base_abs": valuation.fair_value_base_abs,
        "fair_value_bear_abs": valuation.fair_value_bear_abs,
        "fair_value_bull_abs": valuation.fair_value_bull_abs,
    }
    bad = [name for name, (low, high) in ranges.items() if not (low < high)]
    return CheckResult(
        name="ranges_ordered",
        passed=not bad,
        message=f"out-of-order ranges: {bad}" if bad else "ok",
    )


def _check_buy_zone_discount(verdict: VerdictJSON, valuation: ValuationComputed) -> CheckResult:
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


def validate_report(report_text: str, brief: Brief) -> ValidationResult:
    try:
        verdict = extract_verdict_json(report_text)
    except VerdictParseError as exc:
        return ValidationResult(passed=False, failures=[f"verdict_json_parse: {exc}"])

    valuation = compute_valuation(verdict.valuation_inputs)

    checks = [
        _check_confidence_cap(verdict, brief),
        _check_buy_gate(verdict),
        _check_ranges_ordered(verdict, valuation),
        _check_buy_zone_discount(verdict, valuation),
        _check_confidence_vs_missing_data(verdict, brief),
        _check_bear_eps_sanity(verdict, brief),
        _check_bear_adequacy_for_high_multiple(verdict, valuation, brief),
        _check_price_date_fresh(verdict),
        _check_pledge_not_stated_when_unconfirmed(report_text, brief),
        _check_standalone_disclosed(report_text, brief),
        _check_technical_figures_not_recomputed(report_text, brief),
    ]

    failures = [f"{c.name}: {c.message}" for c in checks if not c.passed]
    return ValidationResult(passed=not failures, failures=failures)


def format_validation_errors(result: ValidationResult) -> str:
    lines = ["The previous attempt failed these checks — fix them and resend the FULL report:"]
    lines.extend(f"- {failure}" for failure in result.failures)
    return "\n".join(lines)
