"""Module 8 — Stage 1: Sonnet 5 structured extraction.

Extraction-only: no verdict, no valuation judgment, no sector-adaptation
opinions. Reads the annual report sections and news already fetched into
the Brief and turns them into structured findings Stage 2 can trust were
derived from that text, not invented from general knowledge.

No `temperature` parameter — Sonnet 5 rejects it outright (HTTP 400), not
just ignores it. See PROJECT.md's "LLM API note" section; Prompt 9's
"Temperature 0" instruction predates this API change.

`queries_searched_but_clean` is deliberately NOT a field here — the fetch
layer already knows which adversarial queries came back empty
(Brief.news.queries_empty). Never ask the model to report what code
already knows from its own data.

Red-flag news is capped per adversarial query before being sent (reusing
brief.py's cap_red_flags_per_query) — a heavily-covered company can have
300+ deduped candidates, mostly noise from one broad query, which would
blow past a reasonable input budget for no extraction benefit.
"""

from __future__ import annotations

from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel, Field

from stockbot.brief import cap_red_flags_per_query
from stockbot.config import settings
from stockbot.llm.client import call_anthropic_and_log
from stockbot.models import Brief, RedFlag

MODEL = "claude-sonnet-5"
# Reverted back from claude-haiku-4-5-20251001. The v3 migration moved
# Stage 1 to Haiku for cost (~1/3 of Sonnet's rate), but the recall
# benchmark this same evening (llm/recall_benchmark.py) found Haiku
# missing real governance red flags that were sitting directly in the
# supplied text — most consistently a Rule 11(g) audit-trail disclosure,
# missed on 2 of 3 hand-verified tickers (VMM, JYOTHYLAB), caught on the
# third (BEL). Extraction is the layer this whole pipeline depends on to
# surface exactly these findings; correctness comes before cost here.
# Revisit only with actual recall numbers from the benchmark, not vibes.
#
# No `thinking` parameter here: this extraction task is reading supplied
# text for stated facts, not exercising judgment, so it doesn't need
# extended reasoning — {"type": "adaptive"} would be the correct shape on
# Sonnet 5 if ever added (confirmed elsewhere in this codebase), but there
# has been no live evidence yet that Stage 1 needs it.
#
# max_tokens=8000: real observed Stage 1 output has been 620-4487 tokens
# even on Sonnet, so this has comfortable headroom — but run_stage1 below
# still fails loudly (Stage1Error) if parsing ever comes back empty,
# rather than silently returning None, in case that changes.
MAX_TOKENS = 8000

SYSTEM_PROMPT = """You are a forensic extraction assistant. You will be given \
extracted sections of a company's annual report (auditor's report, key audit \
matters, contingent liabilities, related party transactions - whichever were \
found) and a list of news items: a general 12-month scan, plus results from \
five adversarial searches (SEBI, auditor resignation, promoter pledge, fraud \
investigation, rating downgrade).

Your ONLY job is extraction, not judgment or valuation. For every field:

- If the provided text does not contain enough information to determine a \
field, set it to null (or an empty list) and add a note to extraction_gaps \
explaining what's missing and why. Never guess or infer a plausible-sounding \
value, and never fill a gap from general knowledge about the company.
- auditor_opinion_type must be based on the actual opinion language in the \
Independent Auditor's Report section, if present. If that section was not \
provided or does not contain a clear opinion statement, set this to null and \
note it in extraction_gaps — do not assume "clean" by default just because \
nothing negative was found.
- key_audit_matters and auditor_concerns must each be short, literal \
paraphrases of content actually present in the provided text.
- related_party_amount_cr and contingent_liabilities_cr must be numbers found \
or directly computable from the provided text, in ₹ crore. If multiple \
amounts are given without a clear total, report the range in \
related_party_summary and leave the amount field null rather than picking one.
- red_flags_found: review the provided news items (both the general scan and \
the five adversarial-query results) and select ONLY the ones that describe a \
genuine, specific negative finding — not routine business news, AGM notices, \
earnings-call writeups, or unrelated companies that happened to match a \
broad query. For each one you select, copy its headline, url, published_date, \
and found_by_query EXACTLY as given in the input. Do not invent or alter any \
of these fields, and do not add a red flag that isn't backed by one of the \
provided items.

Output must be valid JSON matching the provided schema."""


class ExtractionResult(BaseModel):
    auditor_opinion_type: Literal["clean", "qualified", "adverse", "disclaimer"] | None = None
    auditor_concerns: list[str] = Field(default_factory=list)
    key_audit_matters: list[str] = Field(default_factory=list)
    related_party_summary: str | None = None
    related_party_amount_cr: float | None = None
    contingent_liabilities_cr: float | None = None
    red_flags_found: list[RedFlag] = Field(default_factory=list)
    extraction_gaps: list[str] = Field(default_factory=list)


def build_user_message(brief: Brief) -> str:
    parts: list[str] = ["## Annual Report Sections"]

    if brief.annual_report.sections:
        for heading, text in brief.annual_report.sections.items():
            parts.append(f"### {heading}\n{text}")
    else:
        parts.append("MISSING: no annual report sections were extracted.")

    parts.append("\n## News")
    if brief.news is None:
        parts.append("MISSING: news fetch failed entirely — no items to review.")
    else:
        parts.append("### General (last 12 months)")
        if brief.news.general:
            for item in brief.news.general:
                parts.append(f"- {item.headline} | {item.url} | {item.published_date.isoformat()}")
        else:
            parts.append("(none found)")

        parts.append("\n### Adversarial red-flag search results")
        parts.append(f"Queries run: {', '.join(brief.news.queries_run)}")
        if brief.news.queries_empty:
            parts.append(
                "Queries with zero results (nothing to review for these): "
                + ", ".join(brief.news.queries_empty)
            )
        capped = cap_red_flags_per_query(brief.news.red_flags, brief.news.queries_run)
        if capped:
            for item in capped:
                parts.append(
                    f"- [{item.found_by_query}] {item.headline} | {item.url} | "
                    f"{item.published_date.isoformat()}"
                )
        else:
            parts.append("(none found)")

    return "\n".join(parts)


class Stage1Error(Exception):
    pass


def run_stage1(
    brief: Brief, client: Anthropic | None = None, max_tokens: int | None = None
) -> tuple[ExtractionResult, dict]:
    client = client or Anthropic(api_key=settings.anthropic_api_key)
    user_message = build_user_message(brief)

    # SYSTEM_PROMPT is byte-identical on every call regardless of ticker —
    # cache it in principle. Confirmed live it never actually caches,
    # though: Sonnet 5's minimum cacheable prompt length is 1,024 tokens
    # and SYSTEM_PROMPT is only ~530 tokens. Every real Stage 1 call has
    # shown cache_creation_input_tokens: 0 — a silent no-op per the docs,
    # not an error. Left in rather than removed: harmless, and it
    # activates automatically if this prompt ever grows past 1,024. ttl
    # matches Stage 2's for consistency, even though it's currently inert
    # here — see verdict.py's HARD_INJECTIONS block for where the 1h TTL
    # actually pays off (real cache_creation_tokens observed there).
    response, cost_inr = call_anthropic_and_log(
        client,
        stage="stage1",
        ticker=brief.ticker.symbol,
        model=MODEL,
        max_tokens=max_tokens or MAX_TOKENS,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ],
        messages=[{"role": "user", "content": user_message}],
        output_format=ExtractionResult,
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_inr": cost_inr,
    }

    if response.parsed_output is None:
        # The SDK's parse() doesn't raise when there's no text block to
        # parse (e.g. the response was cut off mid-thinking before ever
        # emitting JSON) — it just leaves parsed_output as None. Fail
        # loudly here instead of letting a confusing AttributeError
        # surface three calls further down in Stage 2.
        raise Stage1Error(
            f"Stage 1 produced no parseable output (stop_reason={response.stop_reason!r}, "
            f"output_tokens={response.usage.output_tokens}). This call was still billed "
            f"(₹{cost_inr:.2f}) and logged."
        )

    return response.parsed_output, usage
