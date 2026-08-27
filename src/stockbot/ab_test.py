"""ab_test.py — Prompt 17D: compare Stage 1 extraction quality across
providers (Claude Haiku 4.5 vs DeepSeek V4 Flash) on the SAME real
evidence, to decide whether Stage 1 can move to the ~5x cheaper DeepSeek
without losing extraction quality that the verdict gate depends on.

Costs ~nothing: DeepSeek's free signup grant covers this; the Anthropic
side can reuse an existing fixture instead of a fresh Haiku call if one
is available.

Deviation from the literal ask: there is no "saved Brief fixture" format
in this project. fixtures.py (llm/fixtures.py) saves raw LLM *responses*,
never the Brief that was sent as input, and Brief holds several pandas
DataFrames with no existing serialization path. Building that
serialization now, for one A/B script, would be more infrastructure than
this needs. Instead, run_extraction_ab takes an already-assembled Brief —
callers either reuse one they already have or call assemble_brief(ticker)
fresh, which is itself free (NSE/Screener/news only, no LLM).

DeepSeek speaks the OpenAI wire format: a plain REST POST to its
/chat/completions endpoint, not a new SDK — this project already depends
on httpx for exactly this shape of call.

Model ID and pricing verified against
https://api-docs.deepseek.com/quick_start/pricing/ on 2026-08-26 — not a
third-party aggregator. Re-verify before trusting this for a real decision
if much time has passed; DeepSeek's API has changed pricing structure at
least twice in recent history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from anthropic import Anthropic

from stockbot.config import settings
from stockbot.llm.client import call_anthropic_and_log
from stockbot.llm.extract import MAX_TOKENS as ANTHROPIC_MAX_TOKENS
from stockbot.llm.extract import SYSTEM_PROMPT, ExtractionResult, build_user_message
from stockbot.llm.fixtures import save_response_fixture
from stockbot.models import Brief

DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# The actual test (17D): these three findings drove VMM's real Management
# Quality to 6/10, which failed the BUY gate and produced WATCH. If an
# extraction misses any one of them, the cheaper model turns a real,
# recently-listed company with a disabled audit trail into a BUY ON
# CORRECTION -- keyword-matched against the combined text of every
# relevant ExtractionResult field, not a single one, since a real
# extraction might place a finding under auditor_concerns OR
# key_audit_matters depending on phrasing.
VMM_CRITICAL_FINDINGS = {
    "rule_11g_audit_trail": ["11(g)", "audit trail", "audit-trail"],
    "caro_bank_variance": ["caro"],  # combined with a bank/stock-statement mention, checked separately
    "whistleblower_complaints": ["whistle"],
}


@dataclass
class ExtractionAttempt:
    provider: str
    model: str
    result: ExtractionResult | None
    error: str | None = None
    raw_text: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def _call_anthropic(user_message: str, model: str, ticker: str = "AB_TEST") -> ExtractionAttempt:
    # Found live: this function originally called the API directly without
    # log_call() or save_response_fixture() — every real Haiku/Sonnet call
    # made through the A/B/recall-benchmark tooling was invisible to our
    # own cost tracker even though Anthropic billed it for real, and its
    # raw response was lost instead of being reusable later. Now routes
    # through the same choke point (llm/client.py) as extract.run_stage1
    # and verdict.run_stage2, so a call that isn't logged is a call that
    # can't be made — not a pattern to remember to repeat per call site.
    client = Anthropic(api_key=settings.anthropic_api_key)
    try:
        response, _cost_inr = call_anthropic_and_log(
            client,
            stage="stage1",
            ticker=ticker,
            model=model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_message}],
            output_format=ExtractionResult,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic comparison tool, not the production pipeline: report every provider's failure side by side rather than letting one crash the whole comparison
        return ExtractionAttempt(provider="anthropic", model=model, result=None, error=str(exc))

    if response.parsed_output is None:
        return ExtractionAttempt(
            provider="anthropic", model=model, result=None,
            error=f"no parsed output (stop_reason={response.stop_reason!r})",
        )
    return ExtractionAttempt(
        provider="anthropic", model=model, result=response.parsed_output,
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
    )


def _call_deepseek(user_message: str, model: str, ticker: str = "AB_TEST") -> ExtractionAttempt:
    # Same rationale as _call_anthropic's fixture fix: a raw response that
    # fails to parse (malformed JSON, empty body — both seen live tonight)
    # is exactly the kind of thing worth persisting rather than losing,
    # since re-triggering the same failure to debug it would cost another
    # call. DeepSeek is free, so this is about not losing the response,
    # not about avoiding spend.
    api_key = settings.deepseek_api_key
    if not api_key:
        return ExtractionAttempt(
            provider="deepseek", model=model, result=None,
            error="DEEPSEEK_API_KEY is not set in .env — cannot call DeepSeek",
        )

    schema_instruction = (
        "\n\nReturn ONLY a single JSON object matching this exact schema, with no prose "
        "before or after it:\n" + json.dumps(ExtractionResult.model_json_schema(), indent=2)
    )
    try:
        response = httpx.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT + schema_instruction},
                    {"role": "user", "content": user_message},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": ANTHROPIC_MAX_TOKENS,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        save_response_fixture(
            stage="stage1_deepseek",
            ticker=ticker,
            report_text=raw_text,
            stop_reason=data["choices"][0].get("finish_reason", "unknown"),
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_read_input_tokens": usage.get("prompt_cache_hit_tokens", 0),
            },
        )

        parsed = ExtractionResult.model_validate(json.loads(raw_text))
        return ExtractionAttempt(
            provider="deepseek", model=model, result=parsed, raw_text=raw_text,
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
        )
    except Exception as exc:  # noqa: BLE001 - same rationale as _call_anthropic above
        return ExtractionAttempt(provider="deepseek", model=model, result=None, error=str(exc))


def run_extraction_ab(brief: Brief, models: list[tuple[str, str]]) -> list[ExtractionAttempt]:
    """models: [(provider, model_id), ...], e.g.
    [("anthropic", "claude-haiku-4-5-20251001"), ("deepseek", "deepseek-v4-flash")]
    Runs Stage 1 extraction against each, on the SAME brief/user_message."""
    user_message = build_user_message(brief)
    attempts = []
    for provider, model in models:
        if provider == "anthropic":
            attempts.append(_call_anthropic(user_message, model, ticker=brief.ticker.symbol))
        elif provider == "deepseek":
            attempts.append(_call_deepseek(user_message, model, ticker=brief.ticker.symbol))
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
    return attempts


def format_comparison_table(attempts: list[ExtractionAttempt]) -> str:
    fields = list(ExtractionResult.model_fields.keys())
    header = "| Field | " + " | ".join(f"{a.provider}/{a.model}" for a in attempts) + " |"
    separator = "|---|" + "---|" * len(attempts)
    lines = [header, separator]
    for field_name in fields:
        row = [field_name]
        for attempt in attempts:
            if attempt.result is None:
                row.append(f"ERROR: {attempt.error}")
            else:
                value = getattr(attempt.result, field_name)
                text = str(value).replace("\n", " ")
                row.append(text[:200] + ("…" if len(text) > 200 else ""))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def check_critical_findings(result: ExtractionResult) -> dict[str, bool]:
    """VMM's actual test: does this extraction catch the three findings
    that drove its real Management Quality to 6/10? Searched across every
    relevant field, not one, since a real extraction might place a finding
    under auditor_concerns or key_audit_matters depending on phrasing."""
    haystack = " ".join(
        [
            result.auditor_opinion_type or "",
            *result.auditor_concerns,
            *result.key_audit_matters,
            result.related_party_summary or "",
            *result.extraction_gaps,
        ]
    ).lower()
    return {
        "rule_11g_audit_trail": any(kw in haystack for kw in VMM_CRITICAL_FINDINGS["rule_11g_audit_trail"]),
        "caro_bank_variance": "caro" in haystack and ("bank" in haystack or "stock statement" in haystack),
        "whistleblower_complaints": "whistle" in haystack,
    }


def main() -> None:
    import sys

    from stockbot.brief import assemble_brief
    from stockbot.fetch.tickers import load_symbol_table, resolve_ticker
    from stockbot.models import AmbiguousMatch

    sys.stdout.reconfigure(encoding="utf-8")

    ticker_query = sys.argv[1] if len(sys.argv) > 1 else "VMM"
    table = load_symbol_table()
    resolved = resolve_ticker(ticker_query, table)
    if resolved is None or isinstance(resolved, AmbiguousMatch):
        print(f"Could not cleanly resolve {ticker_query!r}: {resolved}")
        sys.exit(1)

    print(f"Assembling brief for {resolved.symbol} (free — no LLM calls)...")
    brief = assemble_brief(resolved)

    attempts = run_extraction_ab(
        brief, [("anthropic", ANTHROPIC_MODEL), ("deepseek", DEEPSEEK_MODEL)]
    )

    print(format_comparison_table(attempts))
    print()

    for attempt in attempts:
        if attempt.result is None:
            print(f"{attempt.provider}/{attempt.model}: FAILED — {attempt.error}")
            continue
        findings = check_critical_findings(attempt.result)
        all_found = all(findings.values())
        print(f"{attempt.provider}/{attempt.model} critical findings: {findings}")
        print(
            f"  -> {'ALL FOUND' if all_found else 'MISSED AT LEAST ONE'} "
            f"— {'recommend' if all_found else 'do NOT recommend'} for this extraction task"
        )


if __name__ == "__main__":
    main()
