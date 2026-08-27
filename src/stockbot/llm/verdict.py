"""Module 9 — Stage 2: verdict against the external master prompt.

The master prompt (prompts/master-stock-analysis-prompt-v3.md) is loaded
verbatim as the system prompt — never paraphrased, summarised, or edited
in code, per the project's own rule that this asset is the user's, not
ours to alter. (v3 replaced v2 to add the closed-world rule, placeholder
tokens, and moving fair-value arithmetic out of the model — see
PROJECT.md's "v3 migration" note.)

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

Three hard injections, layered into the user message rather than edited
into the master prompt file itself. The first two are now also stated
natively by v3 itself (closed-world rule, confidence cap) — kept anyway as
pipeline-level reinforcement, not because v3 is silent on them:
  1. Technical figures are code-computed [FACT], not to be recomputed.
  2. Confidence capped at 7/10 for this pipeline, regardless of the
     model's own assessment (the master prompt's own confidence scale
     goes to 10; this pipeline's cap is stricter and independent of it).
  3. No web search tool is available in this call — the master prompt's
     own Step 0-1 research requirements have already been satisfied by
     the data-fetching pipeline (see the plan's Prompt 10 rationale: the
     master prompt assumes an interactive model with search; this
     pipeline's Opus call doesn't have that tool).
Plus the requirement for the trailing JSON block itself, whose schema is
spelled out in the injection text so the model doesn't have to infer it.
"""

from __future__ import annotations

import json
import re
from datetime import date

from anthropic import Anthropic
from pydantic import BaseModel

from stockbot.brief import (
    format_financials_section,
    format_price_section,
    format_shareholding_section,
)
from stockbot.config import MASTER_PROMPT_PATH, settings
from stockbot.llm.client import call_anthropic_and_log
from stockbot.llm.extract import ExtractionResult
from stockbot.models import Brief

MODEL = "claude-sonnet-5"
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
# Raised to 32000; still not proven sufficient, watch for further
# truncation rather than assuming this is the last raise needed.
MAX_TOKENS = 32000

PIPELINE_CONFIDENCE_CAP = 7

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def effective_confidence_cap(brief: Brief) -> int:
    """Min of the pipeline-wide 7 cap and the brief's data-driven ceiling."""
    return min(PIPELINE_CONFIDENCE_CAP, brief.confidence_ceiling)


def hard_injections(confidence_cap: int) -> str:
    return f"""
IMPORTANT — pipeline constraints, read before writing the report:

1. The technical figures below (SMA50, SMA200, RSI14, support/resistance) \
were computed from real OHLCV data in code. Treat them as [FACT]. Do not \
recompute, estimate, or second-guess them yourself.

2. Confidence is capped at {confidence_cap}/10 for this pipeline, regardless of your own \
assessment of how confident the analysis is. If your assessment would \
otherwise be higher, cap the number at {confidence_cap} but still describe your true \
assessment in the prose.

3. The research described in this prompt's Step 0-1 (annual report, \
quarterly results, shareholding pattern, a 12-month news scan, and the \
mandatory disconfirmation search) has already been performed by a separate \
data-fetching pipeline and is contained entirely in the material below, \
including a structured extraction of the annual report's auditor findings \
and any red flags found in news. You do NOT have a web search tool in this \
call — do not attempt to research further, and do not assume access to any \
live data beyond what is provided below. Treat the provided material as the \
complete evidence base for this analysis; any gap in it is a genuine \
finding to report as MISSING or [UNVERIFIED], never something to fill in \
from general knowledge about the company.

At the very end of your report, after the Final Beginner Summary, include a \
fenced ```json code block containing exactly these fields. The numbers must \
be the SAME as used in the prose above — this is a restatement for the \
pipeline to parse, never a re-derivation:

```json
{{
  "verdict": "BUY|BUY ON CORRECTION|WATCH|SKIP",
  "current_price_abs": <number>,
  "price_date": "<YYYY-MM-DD>",
  "buy_zone_abs": [<low>, <high>],
  "valuation_inputs": {{
    "eps_bear": <number>, "eps_base": <number>, "eps_bull": <number>,
    "multiple_bear": [<low>, <high>],
    "multiple_base": [<low>, <high>],
    "multiple_bull": [<low>, <high>]
  }},
  "confidence": <integer, 1-10 scale per the master prompt's own CONFIDENCE \
section, but capped at {confidence_cap} per constraint #2 above — never write {confidence_cap} as if it \
were the scale's maximum>,
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
  "gates_failed": ["<string>", "..."]
}}
```

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


class VerdictJSON(BaseModel):
    verdict: str
    current_price_abs: float
    price_date: date
    buy_zone_abs: tuple[float, float]
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
    if not MASTER_PROMPT_PATH.exists():
        raise VerdictParseError(
            f"Master prompt not found at {MASTER_PROMPT_PATH} — this is a required "
            f"external asset, not something the pipeline can proceed without."
        )
    return MASTER_PROMPT_PATH.read_text(encoding="utf-8")


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


def build_user_message(
    brief: Brief, extraction: ExtractionResult, extra_instruction: str | None = None
) -> str:
    parts = [
        format_price_section(brief.price, brief.technicals),
        "",
        format_financials_section(brief.financials),
        "",
        format_shareholding_section(brief.shareholding),
    ]
    pledge_note = _pledge_warning(brief)
    if pledge_note:
        parts.append(pledge_note)
    parts.extend(
        [
            "",
            _format_extraction_result(extraction),
            hard_injections(effective_confidence_cap(brief)),
        ]
    )
    if extra_instruction:
        parts.append(f"\n### Retry feedback from the previous attempt\n{extra_instruction}")
    return "\n".join(parts)


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
) -> tuple[str, VerdictJSON, dict]:
    # model override exists only for pre-Opus cost-conscious debugging (e.g.
    # running the real master prompt + hard injections against Sonnet 5,
    # ~5x cheaper, to check the prompt itself works before ever spending on
    # Opus) — production calls always take the default (Opus 5).
    client = client or Anthropic(api_key=settings.anthropic_api_key)
    call_model = model or MODEL
    master_prompt = load_master_prompt()
    user_message = build_user_message(brief, extraction, extra_instruction)

    # Streaming, not .create(): the Anthropic SDK refuses a non-streaming
    # call once max_tokens is high enough that generation could plausibly
    # exceed 10 minutes ("Streaming is required for operations that may
    # take longer than 10 minutes") — hit live once max_tokens reached
    # 32000. get_final_message() returns the same Message shape .create()
    # would have, so nothing below this needs to know the difference.
    response, cost_inr = call_anthropic_and_log(
        client,
        stage="stage2",
        ticker=brief.ticker.symbol,
        model=call_model,
        max_tokens=max_tokens or MAX_TOKENS,
        # master_prompt is byte-identical on every call regardless of
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
            {"type": "text", "text": master_prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ],
        # {"type": "adaptive"} is the only accepted shape on Sonnet 5 —
        # budget_tokens is rejected outright (400), not just deprecated.
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )

    report_text = "".join(block.text for block in response.content if block.type == "text")

    if response.stop_reason == "max_tokens":
        raise TruncatedResponseError(cost_inr, len(report_text), max_tokens or MAX_TOKENS)

    verdict = extract_verdict_json(report_text)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_inr": cost_inr,
    }

    return report_text, verdict, usage
